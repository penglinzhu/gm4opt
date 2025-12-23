# gm4opt_graph.py
# 从 ModelIR 自动构建 GraphIR（当前是一个简单的依赖图 + 自检与统计）

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Set

from gm4opt_ir import (
    MetaInfo,
    SetDef,
    ParamDef,
    VarDef,
    ObjectiveDef,
    ConstraintDef,
    ModelIR,
    GraphIR,
    GraphNode,
    GraphEdge,
)


@dataclass
class GraphCheckIssue:
    """
    图结构自检结果

    kind:     问题类型，例如 "orphan_var", "unused_param", "disconnected_component"
    severity: "info" / "warning" / "error"
    message:  简要说明
    nodes:    相关 node_id 列表，方便定位
    """
    kind: str
    severity: str
    message: str
    nodes: List[str]


def _build_adjacency(nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> Dict[str, Set[str]]:
    """
    构建无向邻接表，用于统计度数和连通分量。
    """
    adj: Dict[str, Set[str]] = {nid: set() for nid in nodes.keys()}
    for e in edges:
        if e.source in adj and e.target in adj:
            adj[e.source].add(e.target)
            adj[e.target].add(e.source)
    return adj


def _compute_graph_stats(nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> Dict[str, Any]:
    """
    基于 nodes / edges 计算一些简单的图统计量：
    - 节点/边数量
    - 各类型节点数量
    - 度数统计
    - 连通分量数量及规模
    """
    n_nodes = len(nodes)
    n_edges = len(edges)

    n_set_nodes = sum(1 for n in nodes.values() if n.type == "set")
    n_param_nodes = sum(1 for n in nodes.values() if n.type == "param")
    n_var_nodes = sum(1 for n in nodes.values() if n.type == "var")
    n_con_nodes = sum(1 for n in nodes.values() if n.type == "constraint")
    n_obj_nodes = sum(1 for n in nodes.values() if n.type == "objective")

    adj = _build_adjacency(nodes, edges)
    degrees = [len(neis) for neis in adj.values()] if adj else [0]
    max_degree = max(degrees) if degrees else 0
    avg_degree = (sum(degrees) / len(degrees)) if degrees and len(degrees) > 0 else 0.0

    # 连通分量
    visited: Set[str] = set()
    component_sizes: List[int] = []

    for nid in nodes.keys():
        if nid in visited:
            continue
        # BFS/DFS 均可，这里用简单 DFS
        stack = [nid]
        size = 0
        visited.add(nid)
        while stack:
            cur = stack.pop()
            size += 1
            for nb in adj.get(cur, []):
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        component_sizes.append(size)

    n_connected_components = len(component_sizes) if component_sizes else 0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_set_nodes": n_set_nodes,
        "n_param_nodes": n_param_nodes,
        "n_var_nodes": n_var_nodes,
        "n_con_nodes": n_con_nodes,
        "n_obj_nodes": n_obj_nodes,
        "max_degree": max_degree,
        "avg_degree": avg_degree,
        "n_connected_components": n_connected_components,
        "component_sizes": component_sizes,
    }


def _run_graph_self_checks(nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> List[GraphCheckIssue]:
    """
    基于已有节点/边信息做一些轻量级自检：
    - orphan_var: 变量只被索引集合连接，但不出现在 objective/constraints 中
    - unused_param: 参数未出现在 objective/constraints 中
    - unused_set: 集合节点既不索引 param/var，也不经由 param/var 连接到任何约束/目标
    - obj_without_var: 目标函数不包含任何变量节点
    - no_constraint: 没有约束节点
    - disconnected_component: 图存在多个连通分量
    """
    issues: List[GraphCheckIssue] = []

    # 分类节点
    var_nodes = [nid for nid, n in nodes.items() if n.type == "var"]
    param_nodes = [nid for nid, n in nodes.items() if n.type == "param"]
    set_nodes = [nid for nid, n in nodes.items() if n.type == "set"]
    con_nodes = [nid for nid, n in nodes.items() if n.type == "constraint"]
    obj_nodes = [nid for nid, n in nodes.items() if n.type == "objective"]

    # 用于判断“在目标/约束中是否用到”
    var_used_in_model: Set[str] = set()
    param_used_in_model: Set[str] = set()

    # 目标中是否出现过变量
    obj_has_var: Dict[str, bool] = {nid: False for nid in obj_nodes}

    # 遍历边，基于 etype 做简单分类
    for e in edges:
        if e.etype in ("var-in-objective", "var-in-constraint"):
            if e.source in var_nodes:
                var_used_in_model.add(e.source)

        if e.etype in ("param-in-objective", "param-in-constraint"):
            if e.source in param_nodes:
                param_used_in_model.add(e.source)

        if e.etype == "var-in-objective" and e.target in obj_has_var:
            obj_has_var[e.target] = True

    # orphan_var: 没有出现在 objective/constraints 中的变量节点
    orphan_vars = [nid for nid in var_nodes if nid not in var_used_in_model]
    if orphan_vars:
        issues.append(
            GraphCheckIssue(
                kind="orphan_var",
                severity="warning",
                message=f"{len(orphan_vars)} variable node(s) do not appear in any objective/constraint.",
                nodes=sorted(orphan_vars),
            )
        )

    # unused_param: 没有出现在 objective/constraints 中的参数节点
    unused_params = [nid for nid in param_nodes if nid not in param_used_in_model]
    if unused_params:
        issues.append(
            GraphCheckIssue(
                kind="unused_param",
                severity="warning",
                message=f"{len(unused_params)} parameter node(s) do not appear in any objective/constraint.",
                nodes=sorted(unused_params),
            )
        )

    # obj_without_var: 目标函数中不包含任何变量（有时是模型错误信号）
    for obj_id, has_var in obj_has_var.items():
        if not has_var:
            issues.append(
                GraphCheckIssue(
                    kind="obj_without_var",
                    severity="warning",
                    message=f"Objective node {obj_id} does not reference any variable node.",
                    nodes=[obj_id],
                )
            )

    # no_constraint: 没有任何约束节点
    if len(con_nodes) == 0:
        issues.append(
            GraphCheckIssue(
                kind="no_constraint",
                severity="warning",
                message="Graph contains no constraint nodes.",
                nodes=[],
            )
        )

    # unused_set: 与任何 param/var 无连接的集合节点
    adj = _build_adjacency(nodes, edges)
    unused_sets = [nid for nid in set_nodes if len(adj.get(nid, [])) == 0]
    if unused_sets:
        issues.append(
            GraphCheckIssue(
                kind="unused_set",
                severity="info",
                message=f"{len(unused_sets)} set node(s) are not connected to any param/var.",
                nodes=sorted(unused_sets),
            )
        )

    # disconnected_component: 连通分量数 > 1
    visited: Set[str] = set()
    component_sizes: List[int] = []
    components_nodes: List[List[str]] = []

    for nid in nodes.keys():
        if nid in visited:
            continue
        stack = [nid]
        comp_nodes: List[str] = []
        visited.add(nid)
        while stack:
            cur = stack.pop()
            comp_nodes.append(cur)
            for nb in adj.get(cur, []):
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        component_sizes.append(len(comp_nodes))
        components_nodes.append(comp_nodes)

    n_components = len(component_sizes)
    if n_components > 1:
        # 这里只给一条 issue，把各个分量节点列表放到 nodes
        flat_nodes: List[str] = []
        for comp in components_nodes:
            flat_nodes.extend(comp)
        issues.append(
            GraphCheckIssue(
                kind="disconnected_component",
                severity="info",
                message=f"Graph has {n_components} connected components.",
                nodes=flat_nodes,
            )
        )

    return issues


def build_graph_from_ir(ir: ModelIR) -> GraphIR:
    """
    根据 ModelIR 构造一个简单的 GraphIR：
    - 每个 set/param/var/objective/constraint 都变成一个节点
    - 根据表达式里的名字，建立依赖边：
        * set -> param/var （索引关系）
        * param/var -> objective
        * param/var -> constraint
    - 构造完成后，进行一次自检（结构/使用情况）并统计若干图特征，
      写入 GraphIR.meta["checks"] 与 ["stats"]。

    """
    nodes: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []

    def add_node(node_id: str, ntype: str, label: str | None = None, attrs=None):
        if node_id not in nodes:
            nodes[node_id] = GraphNode(
                id=node_id,
                type=ntype,
                label=label if label is not None else node_id,
                attrs=attrs,
            )

    def add_edge(src: str, dst: str, etype: str, attrs=None):
        edges.append(
            GraphEdge(
                source=src,
                target=dst,
                etype=etype,
                attrs=attrs,
            )
        )

    # ---- 1. sets → 节点 ----
    for s in ir.sets:
        node_id = f"set:{s.name}"
        add_node(
            node_id=node_id,
            ntype="set",
            label=s.name,
            attrs={"elements": list(s.elements)},
        )

    # ---- 2. params → 节点 + set -> param 边 ----
    for p in ir.params:
        node_id = f"param:{p.name}"
        add_node(
            node_id=node_id,
            ntype="param",
            label=p.name,
            attrs={"indices": list(p.indices)},
        )
        # 索引集与参数之间的连边
        for set_name in p.indices:
            set_node_id = f"set:{set_name}"
            if set_node_id in nodes:
                add_edge(set_node_id, node_id, etype="set-index-of-param")

    # ---- 3. vars → 节点 + set -> var 边 ----
    for v in ir.vars:
        node_id = f"var:{v.name}"
        add_node(
            node_id=node_id,
            ntype="var",
            label=v.name,
            attrs={"indices": list(v.indices), "vartype": v.vartype},
        )
        for set_name in v.indices:
            set_node_id = f"set:{set_name}"
            if set_node_id in nodes:
                add_edge(set_node_id, node_id, etype="set-index-of-var")

    # ---- 4. objective → 节点 ----
    obj = ir.objective
    obj_node_id = f"obj:{obj.name}"
    add_node(
        node_id=obj_node_id,
        ntype="objective",
        label=obj.name,
        attrs={"sense": obj.sense, "expr": obj.expr},
    )

    # ---- 5. constraints → 节点 ----
    for c in ir.constraints:
        cons_node_id = f"con:{c.name}"
        add_node(
            node_id=cons_node_id,
            ntype="constraint",
            label=c.name,
            attrs={
                "sense": c.sense,
                "expr_lhs": c.expr_lhs,
                "expr_rhs": c.expr_rhs,
            },
        )

    # ---- 6. 基于字符串的依赖边 ----

    var_names = [v.name for v in ir.vars]
    param_names = [p.name for p in ir.params]

    # Objective 依赖：var/param -> objective
    obj_expr = obj.expr or ""
    for vname in var_names:
        if vname in obj_expr:
            add_edge(f"var:{vname}", obj_node_id, etype="var-in-objective")
    for pname in param_names:
        if pname in obj_expr:
            add_edge(f"param:{pname}", obj_node_id, etype="param-in-objective")

    # Constraints 依赖：var/param -> constraint
    for c in ir.constraints:
        cons_node_id = f"con:{c.name}"
        full_expr = f"{c.expr_lhs} ; {c.expr_rhs}"
        for vname in var_names:
            if vname in full_expr:
                add_edge(f"var:{vname}", cons_node_id, etype="var-in-constraint")
        for pname in param_names:
            if pname in full_expr:
                add_edge(f"param:{pname}", cons_node_id, etype="param-in-constraint")

    # ---- 7. 构造 GraphIR，并进行自检与统计 ----
    stats = _compute_graph_stats(nodes, edges)
    issues = _run_graph_self_checks(nodes, edges)

    graph_meta: Dict[str, Any] = {
        "builder": "build_graph_from_ir",
        "version": 1,
        "stats": stats,
        "checks": {
            "issues": [asdict(iss) for iss in issues],
        },
    }

    return GraphIR(nodes=nodes, edges=edges, meta=graph_meta)
