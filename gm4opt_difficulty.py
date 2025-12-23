# gm4opt_difficulty.py
# 基于 ModelIR + GraphIR 的简单 difficulty-aware 启发式（以图结构为主）

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from gm4opt_ir import ModelIR


@dataclass
class DifficultyEstimate:
    """
    对单个 ModelIR 的难度估计结果。

    - score: 一个连续的难度分数（越大越难）
    - level: "easy" / "medium" / "hard" 等粗粒度标签
    - features: 用于估计的特征字典
    """
    score: float
    level: str
    features: Dict[str, Any]


def estimate_difficulty(ir: ModelIR) -> DifficultyEstimate:
    """
    根据 ModelIR（尤其是 GraphIR）估计一个难度分数。
    """
    #1. IR基础计数
    n_sets = len(ir.sets)
    n_params = len(ir.params)
    n_vars = len(ir.vars)
    n_constraints = len(ir.constraints)

    # 各类变量数量
    n_binary_vars = sum(1 for v in ir.vars if v.vartype == "binary")
    n_integer_vars = sum(1 for v in ir.vars if v.vartype == "integer")
    n_continuous_vars = n_vars - n_binary_vars - n_integer_vars

    # 变量/参数维度信息
    n_2d_vars = sum(1 for v in ir.vars if len(v.indices) == 2)
    n_2d_params = sum(1 for p in ir.params if len(p.indices) == 2)

    #2. GraphIR相关特征
    n_graph_nodes = 0
    n_graph_edges = 0
    n_connected_components = 0
    n_orphan_vars = 0
    n_unused_params = 0
    n_unused_sets = 0

    if ir.graph is not None:
        stats = ir.graph.meta.get("stats", {}) if ir.graph.meta else {}
        n_graph_nodes = stats.get("n_nodes", len(ir.graph.nodes))
        n_graph_edges = stats.get("n_edges", len(ir.graph.edges))
        n_connected_components = stats.get("n_connected_components", 0)

        checks = ir.graph.meta.get("checks", {}) if ir.graph.meta else {}
        issues = checks.get("issues", [])

        for iss in issues:
            kind = iss.get("kind", "")
            nodes = iss.get("nodes", []) or []
            if kind == "orphan_var":
                n_orphan_vars += len(nodes)
            elif kind == "unused_param":
                n_unused_params += len(nodes)
            elif kind == "unused_set":
                n_unused_sets += len(nodes)
            elif kind == "disconnected_component":
                pass

    #3. 汇总特征
    features: Dict[str, Any] = {
        "n_sets": n_sets,
        "n_params": n_params,
        "n_vars": n_vars,
        "n_constraints": n_constraints,
        "n_binary_vars": n_binary_vars,
        "n_integer_vars": n_integer_vars,
        "n_continuous_vars": n_continuous_vars,
        "n_2d_vars": n_2d_vars,
        "n_2d_params": n_2d_params,
        "n_graph_nodes": n_graph_nodes,
        "n_graph_edges": n_graph_edges,
        "n_connected_components": n_connected_components,
        "n_orphan_vars": n_orphan_vars,
        "n_unused_params": n_unused_params,
        "n_unused_sets": n_unused_sets,
    }

    #4. 线性打分
    score = 0.0

    # 基本结构复杂度（规模越大越难）
    score += 0.5 * n_sets
    score += 0.8 * n_params
    score += 1.0 * n_vars
    score += 0.7 * n_constraints

    # 变量类型影响：整数/二进制更难
    score += 2.0 * n_integer_vars
    score += 3.0 * n_binary_vars

    # 高维结构带来的额外复杂度
    score += 1.0 * n_2d_vars
    score += 0.5 * n_2d_params

    # 图结构规模：轻微加权
    score += 0.05 * n_graph_nodes
    score += 0.02 * n_graph_edges

    # 图的拓扑复杂度：多个连通分量通常表示模型结构较碎、依赖关系更复杂
    if n_connected_components > 1:
        score += 2.0 * (n_connected_components - 1)

    # 建模质量问题：orphan_var / unused_param / unused_set
    score += 1.0 * n_orphan_vars
    score += 0.5 * n_unused_params
    score += 0.3 * n_unused_sets

    #5. 粗粒度difficulty level
    # 阈值目前是随便设置的
    if score < 10:
        level = "easy"
    elif score < 25:
        level = "medium"
    else:
        level = "hard"

    return DifficultyEstimate(score=score, level=level, features=features)


def attach_difficulty(ir: ModelIR) -> DifficultyEstimate:
    """
    对 IR 做难度估计，并把结果挂在 IR 上。
    """
    est = estimate_difficulty(ir)

    # 把结果挂到 graph.meta 里，方便后续使用
    if ir.graph is not None:
        if ir.graph.meta is None:
            ir.graph.meta = {}
        ir.graph.meta["difficulty"] = {
            "score": est.score,
            "level": est.level,
            "features": est.features,
        }

    return est
