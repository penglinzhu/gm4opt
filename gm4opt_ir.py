# gm4opt_ir.py
# GM4OPT IR dataclasses + IR -> Gurobi model builder

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import gurobipy as gp
from gurobipy import GRB


# ========= IR Dataclasses =========

@dataclass
class MetaInfo:
    """Metadata about the model/problem."""
    problem_id: str
    source: Optional[str] = None
    description: Optional[str] = None
    sense: str = "min"  # "min" or "max"
    version: int = 0


@dataclass
class SetDef:
    """Finite set definition, e.g., WORKERS, TASKS."""
    name: str
    elements: List[Any]
    description: Optional[str] = None


@dataclass
class ParamDef:
    """
    Model parameter, e.g., cost[c], time[w][t].

    - 0D: indices = [], values 可以是数字 (int/float)，也可以是只含一个键的 dict。
    - 1D: indices = ["SET"], values 是 dict: {key -> value}.
    - 2D: indices = ["SET1","SET2"], values 是嵌套 dict:
        {
          i1: {j1: v_11, j2: v_12, ...},
          i2: {j1: v_21, ...},
          ...
        }
    """
    name: str
    indices: List[str]
    values: object | None = None
    description: Optional[str] = None


@dataclass
class VarDef:
    """
    Decision variable.

    - 0D: indices = []         -> scalar variable
    - 1D: indices = ["SET"]    -> x[i]
    - 2D: indices = ["S1","S2"]-> x[i][j]
    """
    name: str
    indices: List[str]
    vartype: str  # "binary", "integer", "continuous"
    lb: float = 0.0
    ub: Optional[float] = None
    description: Optional[str] = None


@dataclass
class ObjectiveDef:
    """Objective definition with a Python expression string."""
    name: str
    sense: str  # "min" or "max"
    expr: str
    description: Optional[str] = None


@dataclass
class ConstraintDef:
    """
    Constraint definition.

    IMPORTANT: In this version, each constraint MUST be a fully specified
    scalar constraint. There must be NO free index variables like c, t, w.
    """
    name: str
    expr_lhs: str
    sense: str  # "<=", ">=", "=="
    expr_rhs: str
    description: Optional[str] = None


# ========= Graph IR Dataclasses (for GM4OPT) =========

@dataclass
class GraphNode:
    """
    Node in the GM4OPT graph IR.

    id:   unique node id in the graph (string).
    type: semantic type, e.g. "set", "param", "var", "constraint", "objective".
    label: human-readable label (optional).
    attrs: arbitrary extra attributes (optional, can be used later for analysis).
    """
    id: str
    type: str
    label: Optional[str] = None
    attrs: Optional[Dict[str, Any]] = None


@dataclass
class GraphEdge:
    """
    Edge in the GM4OPT graph IR.

    source / target: node ids.
    etype: semantic type of the edge, e.g. "uses", "defined_on", "appears_in".
    attrs: arbitrary extra attributes (optional).
    """
    source: str
    target: str
    etype: str
    attrs: Optional[Dict[str, Any]] = None


@dataclass
class GraphIR:
    """
    Graph-based intermediate representation of a model.

    - nodes: mapping from node_id to GraphNode
    - edges: list of GraphEdge
    - meta:  optional metadata about the graph itself (e.g., construction version)
    """
    nodes: Dict[str, GraphNode]
    edges: List[GraphEdge]
    meta: Optional[Dict[str, Any]] = None


@dataclass
class ModelIR:
    """Full IR for a model."""
    meta: MetaInfo
    sets: List[SetDef]
    params: List[ParamDef]
    vars: List[VarDef]
    objective: ObjectiveDef
    constraints: List[ConstraintDef]
    # Graph IR is the core novelty of GM4OPT; None if not constructed / disabled.
    graph: Optional[GraphIR] = None


# ========= IR -> Gurobi =========

def ir_to_gurobi(ir: ModelIR) -> gp.Model:
    """
    Convert ModelIR to a Gurobi model.

    Assumes:
    - All constraints are scalar and directly evaluable (no free indices).
    - Expressions are valid Python expressions using:
        - sets by their names,
        - params by their names (0/1/2D),
        - vars by their names (0/1/2D),
        - functions: sum, quicksum, enumerate, range, len, max, min, abs.
    """

    # ---- 0. Create model ----
    model_name = ir.meta.problem_id or "gm4opt_model"
    m = gp.Model(model_name)

    # ---- 1. Sets ----
    env_sets: Dict[str, List[Any]] = {}
    eval_env: Dict[str, Any] = {}

    for s in ir.sets:
        elements = list(s.elements)
        env_sets[s.name] = elements
        eval_env[s.name] = elements

    # ---- 2. Params ----
    env_params: Dict[str, Any] = {}

    for p in ir.params:
        # 0D param (scalar)
        if not p.indices:
            val = p.values
            if isinstance(val, (int, float)):
                env_params[p.name] = float(val)
            elif isinstance(val, dict):
                if len(val) != 1:
                    raise ValueError(
                        f"Scalar param {p.name} expects exactly one value in dict, "
                        f"got keys={list(val.keys())}."
                    )
                env_params[p.name] = float(next(iter(val.values())))
            elif val is None:
                raise ValueError(f"Scalar param {p.name} has no value (None).")
            else:
                raise TypeError(
                    f"Scalar param {p.name} has unsupported values type: {type(val)}; "
                    f"expected number or single-entry dict."
                )

        # 1D param
        elif len(p.indices) == 1:
            if p.values is None:
                env_params[p.name] = {}
            elif isinstance(p.values, dict):
                env_params[p.name] = dict(p.values)
            else:
                raise TypeError(
                    f"1D param {p.name} expects dict for values, got {type(p.values)}."
                )

        # 2D param
        elif len(p.indices) == 2:
            if p.values is None:
                env_params[p.name] = {}
            elif isinstance(p.values, dict):
                outer: Dict[Any, Dict[Any, float]] = {}
                for i, inner in p.values.items():
                    if not isinstance(inner, dict):
                        raise TypeError(
                            f"2D param {p.name} expects nested dict for values[{i}], "
                            f"got {type(inner)}."
                        )
                    outer[i] = dict(inner)
                env_params[p.name] = outer
            else:
                raise TypeError(
                    f"2D param {p.name} expects nested dict for values, "
                    f"got {type(p.values)}."
                )
        else:
            raise NotImplementedError("GM4OPT v1 only supports up to 2D params.")

    for name, val in env_params.items():
        eval_env[name] = val

    # ---- 3. Vars ----
    env_vars: Dict[str, Any] = {}

    for v in ir.vars:
        # Map vartype string -> Gurobi type
        if v.vartype == "binary":
            vtype = GRB.BINARY
        elif v.vartype == "integer":
            vtype = GRB.INTEGER
        else:
            vtype = GRB.CONTINUOUS

        # 0D var
        if not v.indices:
            var = m.addVar(lb=v.lb, ub=v.ub if v.ub is not None else GRB.INFINITY,
                           vtype=vtype, name=v.name)
            env_vars[v.name] = var

        # 1D var
        elif len(v.indices) == 1:
            set_name = v.indices[0]
            if set_name not in env_sets:
                raise KeyError(
                    f"Var {v.name} refers to set {set_name}, which is not defined."
                )
            idx_list = env_sets[set_name]
            v_dict: Dict[Any, gp.Var] = {}
            for i in idx_list:
                var = m.addVar(
                    lb=v.lb,
                    ub=v.ub if v.ub is not None else GRB.INFINITY,
                    vtype=vtype,
                    name=f"{v.name}[{i}]",
                )
                v_dict[i] = var
            env_vars[v.name] = v_dict

        # 2D var
        elif len(v.indices) == 2:
            s1, s2 = v.indices
            if s1 not in env_sets or s2 not in env_sets:
                raise KeyError(
                    f"Var {v.name} refers to sets {v.indices}, which are not fully defined."
                )
            idx1 = env_sets[s1]
            idx2 = env_sets[s2]
            v_dict2: Dict[Any, Dict[Any, gp.Var]] = {}
            for i in idx1:
                inner: Dict[Any, gp.Var] = {}
                for j in idx2:
                    var = m.addVar(
                        lb=v.lb,
                        ub=v.ub if v.ub is not None else GRB.INFINITY,
                        vtype=vtype,
                        name=f"{v.name}[{i},{j}]",
                    )
                    inner[j] = var
                v_dict2[i] = inner
            env_vars[v.name] = v_dict2

        else:
            raise NotImplementedError("GM4OPT v1 only supports up to 2D vars.")

    for name, val in env_vars.items():
        eval_env[name] = val

    # ---- 4. Functions allowed in expressions ----
    eval_env["quicksum"] = gp.quicksum

    SAFE_FUNCS = {
        "sum": sum,
        "enumerate": enumerate,
        "range": range,
        "len": len,
        "max": max,
        "min": min,
        "abs": abs,
    }
    eval_env.update(SAFE_FUNCS)

    # ---- Global eval environment ----
    global_env: Dict[str, Any] = {"__builtins__": {}}
    global_env.update(eval_env)

    # ---- 5. Objective ----
    obj_def = ir.objective
    print("[GM4OPT DEBUG] eval_env keys for objective:", sorted(eval_env.keys()))
    print("[GM4OPT DEBUG] objective expr:", obj_def.expr)

    obj_expr = eval(obj_def.expr, global_env, {})

    if obj_def.sense.lower() == "min":
        m.setObjective(obj_expr, GRB.MINIMIZE)
    else:
        m.setObjective(obj_expr, GRB.MAXIMIZE)

    # ---- 6. Constraints ----
    # IMPORTANT: We assume all constraints are scalar and directly evaluable.
    # There is NO automatic index expansion logic here.
    for cons_def in ir.constraints:
        lhs = eval(cons_def.expr_lhs, global_env, {})
        rhs = eval(cons_def.expr_rhs, global_env, {})

        if cons_def.sense == "<=":
            m.addConstr(lhs <= rhs, name=cons_def.name)
        elif cons_def.sense == ">=":
            m.addConstr(lhs >= rhs, name=cons_def.name)
        elif cons_def.sense == "==":
            m.addConstr(lhs == rhs, name=cons_def.name)
        else:
            raise ValueError(
                f"Unknown constraint sense '{cons_def.sense}' in {cons_def.name}."
            )

    m.update()
    return m
