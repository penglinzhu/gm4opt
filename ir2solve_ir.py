# ir2solve_ir.py
# ModelIR dataclasses + IR -> Gurobi builder

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import gurobipy as gp
from gurobipy import GRB


# -----------------------------------------------------------------------------
# Core IR
# -----------------------------------------------------------------------------
@dataclass
class MetaInfo:
    problem_id: str
    source: Optional[str] = None
    description: Optional[str] = None
    sense: str = "min"  # "min" | "max"
    version: int = 1


@dataclass
class SetDef:
    name: str
    elements: List[Any]
    description: Optional[str] = None


@dataclass
class ParamDef:
    name: str
    indices: List[str]  # [], [S], or [S1,S2]
    values: Any = None
    description: Optional[str] = None


@dataclass
class VarDef:
    name: str
    indices: List[str]  # [], [S], or [S1,S2]
    vartype: str        # "binary" | "integer" | "continuous"
    lb: float = 0.0
    ub: Optional[float] = None
    description: Optional[str] = None


@dataclass
class ObjectiveDef:
    name: str
    sense: str          # "min" | "max"
    expr: str
    description: Optional[str] = None


@dataclass
class ConstraintDef:
    name: str
    expr_lhs: str
    sense: str          # "<=" | ">=" | "=="
    expr_rhs: str
    description: Optional[str] = None


@dataclass
class ModelIR:
    meta: MetaInfo
    sets: List[SetDef]
    params: List[ParamDef]
    vars: List[VarDef]
    objective: ObjectiveDef
    constraints: List[ConstraintDef]


# -----------------------------------------------------------------------------
# IR -> Gurobi
# -----------------------------------------------------------------------------
def ir_to_gurobi(ir: ModelIR) -> gp.Model:
    """
    Build a Gurobi model from ModelIR.

    Assumptions (kept intentionally strict and simple):
      - constraints are scalar expressions (already unrolled)
      - expressions use Python syntax and only reference:
          sets, params, vars, and safe helper functions
    """
    m = gp.Model(ir.meta.problem_id or "ir2solve_model")

    # 1) sets
    env_sets: Dict[str, List[Any]] = {s.name: list(s.elements) for s in (ir.sets or [])}

    # 2) params (support 0/1/2D)
    env_params: Dict[str, Any] = {}
    for p in ir.params or []:
        if not (p.indices or []):  # scalar
            v = p.values
            if isinstance(v, dict) and len(v) == 1:
                v = next(iter(v.values()))
            if not isinstance(v, (int, float)):
                raise TypeError(f"Scalar param '{p.name}' must be a number (or single-entry dict).")
            env_params[p.name] = float(v)

        elif len(p.indices) == 1:
            if p.values is None:
                env_params[p.name] = {}
            elif isinstance(p.values, dict):
                env_params[p.name] = dict(p.values)
            else:
                raise TypeError(f"1D param '{p.name}' must be a dict.")

        elif len(p.indices) == 2:
            if p.values is None:
                env_params[p.name] = {}
            elif isinstance(p.values, dict):
                # accept either nested dict or tuple-key dict (verifier may normalize)
                env_params[p.name] = p.values
            else:
                raise TypeError(f"2D param '{p.name}' must be a dict (nested or tuple-key).")

        else:
            raise NotImplementedError("Params >2D are not supported.")

    # 3) vars (0/1/2D)
    def _vtype(vt: str):
        if vt == "binary":
            return GRB.BINARY
        if vt == "integer":
            return GRB.INTEGER
        return GRB.CONTINUOUS

    env_vars: Dict[str, Any] = {}
    for v in ir.vars or []:
        vt = _vtype(v.vartype)
        ub = v.ub if v.ub is not None else GRB.INFINITY

        if not (v.indices or []):
            env_vars[v.name] = m.addVar(lb=v.lb, ub=ub, vtype=vt, name=v.name)

        elif len(v.indices) == 1:
            s = v.indices[0]
            if s not in env_sets:
                raise KeyError(f"Var '{v.name}' refers to undefined set '{s}'.")
            d: Dict[Any, gp.Var] = {}
            for i in env_sets[s]:
                d[i] = m.addVar(lb=v.lb, ub=ub, vtype=vt, name=f"{v.name}[{i}]")
            env_vars[v.name] = d

        elif len(v.indices) == 2:
            s1, s2 = v.indices
            if s1 not in env_sets or s2 not in env_sets:
                raise KeyError(f"Var '{v.name}' refers to undefined sets {v.indices}.")
            d2: Dict[Any, Dict[Any, gp.Var]] = {}
            for i in env_sets[s1]:
                inner: Dict[Any, gp.Var] = {}
                for j in env_sets[s2]:
                    inner[j] = m.addVar(lb=v.lb, ub=ub, vtype=vt, name=f"{v.name}[{i},{j}]")
                d2[i] = inner
            env_vars[v.name] = d2

        else:
            raise NotImplementedError("Vars >2D are not supported.")

    # 4) eval env (safe)
    eval_env: Dict[str, Any] = {}
    eval_env.update(env_sets)
    eval_env.update(env_params)
    eval_env.update(env_vars)
    eval_env["quicksum"] = gp.quicksum
    eval_env.update(
        {"sum": sum, "enumerate": enumerate, "range": range, "len": len, "max": max, "min": min, "abs": abs}
    )

    global_env: Dict[str, Any] = {"__builtins__": {}}
    global_env.update(eval_env)

    # 5) objective
    obj = eval(ir.objective.expr, global_env, {})
    m.setObjective(obj, GRB.MINIMIZE if ir.objective.sense.lower() == "min" else GRB.MAXIMIZE)

    # 6) constraints
    for c in ir.constraints or []:
        lhs = eval(c.expr_lhs, global_env, {})
        rhs = eval(c.expr_rhs, global_env, {})
        if c.sense == "<=":
            m.addConstr(lhs <= rhs, name=c.name)
        elif c.sense == ">=":
            m.addConstr(lhs >= rhs, name=c.name)
        elif c.sense == "==":
            m.addConstr(lhs == rhs, name=c.name)
        else:
            raise ValueError(f"Unknown constraint sense '{c.sense}' in '{c.name}'.")

    m.update()
    return m
