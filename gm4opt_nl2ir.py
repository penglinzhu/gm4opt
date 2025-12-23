# gm4opt_nl2ir.py
# NL -> JSON IR text -> ModelIR 解析模块


from __future__ import annotations

import json
import re
from dataclasses import fields
from typing import Any, Dict, List, Optional

from gm4opt_ir import (
    MetaInfo,
    SetDef,
    ParamDef,
    VarDef,
    ObjectiveDef,
    ConstraintDef,
    ModelIR,
)

# ========= Prompts =========

BASE_SYSTEM_PROMPT = """
You are an expert in mathematical modeling and optimization.
Your task is to read a natural language description of a linear / integer
optimization problem and output a JSON object describing the model in a
strictly structured Intermediate Representation (IR) called ModelIR.

You MUST:
- Produce valid JSON (no comments, no trailing commas).
- Follow the given schema exactly.
- Ensure that all constraints are fully specified scalar constraints
  (NO free index variables, NO implicit "for all" quantifiers).
""".strip()

SYSTEM_PROMPT = BASE_SYSTEM_PROMPT

SCHEMA_AND_INSTRUCTIONS = r"""
You must generate a JSON object with the following top-level keys:

{
  "meta": { ... },
  "sets": [ ... ],
  "params": [ ... ],
  "vars": [ ... ],
  "objective": { ... },
  "constraints": [ ... ],
  "graph": null
}

1. meta
---------
meta = {
  "problem_id": str,               
  "source": null or str,
  "description": str,
  "sense": "min" or "max",         
  "version": 1
}

2. sets
---------
Each set is:

{
  "name": "CONTAINERS",
  "elements": [...],
  "description": "..."
}

Index consistency for sets/params/vars/expressions (IMPORTANT):
- All set elements that are used as indices in params/vars/expressions
  MUST be strings, even if they look like numbers.

3. params
---------
Each parameter is:

{
  "name": "time",
  "indices": ["WORKERS", "TASKS"], # 0D: [], 1D: ["SET"], 2D: ["SET1","SET2"]
  "values": { ... },               
  "description": "..."
}

Dimensionality rules:
- 0D (scalar): indices=[], values is a number
- 1D: indices=["SET"], values is {key:value}
- 2D: indices=["SET1","SET2"], values is nested dict values[i][j]

Param usage in expressions (0D scalar) (IMPORTANT):
- If a parameter is 0D (scalar, "indices": []), refer to it DIRECTLY by name.

4. vars
---------
Each variable is:

{
  "name": "x",
  "indices": ["WORKERS", "TASKS"], # 0D, 1D, or 2D
  "vartype": "binary" | "integer" | "continuous",
  "lb": 0.0,
  "ub": null or number,
  "description": "..."
}

VERY IMPORTANT: How to choose vartype (be realistic)
----------------------------------------------------
Default rule (STRICT):
- If the decision represents a COUNT of indivisible things, use INTEGER (or BINARY).
  Examples: number of items/people/vehicles/trucks/machines/batches/orders/trips/servings,
  whether to select/assign/open/build (yes/no).
- Use BINARY for yes/no selection/assignment/on-off decisions.

Only use CONTINUOUS when the statement CLEARLY allows fractional amounts or the quantity
is naturally divisible.
Common signals for CONTINUOUS:
- "fractional", "can be split", "divisible", "real-valued", "continuous",
  "any amount", "arbitrary amount", "portion can be fractional", "half", "0.5".
- Physical flows/amounts: flow/shipment/transport amount, kilograms (kg), tons, liters,
  meters, hours of work, energy, money, probability, percentage, data rate (GB/s).
Special note for diet/meal/food:
- If the variable is number of servings/portions/meals/items purchased, default INTEGER
  unless the statement explicitly allows fractional servings/partial portions.

5. objective
---------
{
  "name": "min_total_cost",
  "sense": "min" or "max",
  "expr": "Python expression using sets/params/vars",
  "description": "..."
}

6. constraints (VERY IMPORTANT)
---------
Each constraint is an object:

{
  "name": "constraint_name",
  "expr_lhs": "Python expression (scalar)",
  "sense": "<=" | ">=" | "==",
  "expr_rhs": "Python expression (scalar)",
  "description": "..."
}

CRITICAL RULES FOR CONSTRAINTS:
(1) Every constraint MUST be a fully specified SCALAR constraint.
(2) In constraint expressions, you MUST NOT use Python "for" loops or generator expressions.
(3) If a conceptual constraint holds "for each element", you MUST UNROLL it into one constraint per element.
"""

# ========= Dynamic prompt templating (heuristic) =========

def _simple_text_features(question_text: str) -> Dict[str, Any]:
    txt = (question_text or "")
    lower = txt.lower()

    has_table = ("table" in lower) or ("\\begin{tabular" in lower) or ("|" in txt and "\n" in txt)
    has_for_each = any(kw in lower for kw in ["for each", "for every", "for all", "∀"])
    has_capacity = any(kw in lower for kw in ["capacity", "at most", "no more than", "at least", "<=", ">="])
    has_implication = any(kw in lower for kw in ["if ", "then", "must", "cannot", "won't", "only if", "requires"])
    has_assignment = any(kw in lower for kw in ["assignment", "assign", "task", "worker", "machine", "shift"])

    # flow/transport words
    has_flow = any(kw in lower for kw in [
        "warehouse", "shipping", "shipment", "transport", "supply", "demand", "flow",
        "network", "arc", "edge", "node", "capacity", "gigabytes", "gb/s", "data rate",
        "transfer", "redistribute", "reallocate", "transship"
    ])

    has_routing = any(kw in lower for kw in ["vehicle routing", "vrp", "route", "depot", "customer", "tour"])
    has_production = any(kw in lower for kw in ["produce", "production", "manufacture", "factory", "product", "profit"])

    # explicit fractional/continuous permission
    allows_fraction = any(kw in lower for kw in [
        "fraction", "fractional", "half", "0.5", "can be split", "divisible",
        "continuous", "real-valued", "arbitrary amount", "any amount", "partial portion",
    ])

    # units that often imply divisible quantity
    has_divisible_units = any(kw in lower for kw in [
        "kg", "kilogram", "grams", " g", "ton", "tons", "litre", "liter", "liters",
        "hours", "hour", "mwh", "kwh", "energy", "flow", "gb/s", "gigabytes per second",
        "rate", "volume", "amount of"
    ])

    is_diet_food = any(kw in lower for kw in ["dietitian", "meal plan", "serving", "servings", "portion", "food item", "nutritional"])

    # rebalancing / inventory redistribution signals (key for >= vs ==)
    has_initial_stock = any(kw in lower for kw in [
        "initial stock", "starts with", "starting stock", "initial amount", "initial supply", "initial inventory"
    ])
    has_required_need = any(kw in lower for kw in [
        "needs", "need", "required", "requirement", "must have", "at least", "fulfill", "meet the needs", "meet the requirement"
    ])
    has_redistribution = any(kw in lower for kw in [
        "redistribute", "reallocate", "transfer", "move", "ship", "transport", "send from", "from clinic", "to clinic"
    ])

    n_tokens = len(txt.split())
    size = "small" if n_tokens < 120 else ("medium" if n_tokens < 300 else "large")

    return {
        "has_table": has_table,
        "has_for_each": has_for_each,
        "has_capacity": has_capacity,
        "has_implication": has_implication,
        "has_assignment": has_assignment,
        "has_flow": has_flow,
        "has_routing": has_routing,
        "has_production": has_production,
        "allows_fraction": allows_fraction,
        "has_divisible_units": has_divisible_units,
        "is_diet_food": is_diet_food,
        "has_initial_stock": has_initial_stock,
        "has_required_need": has_required_need,
        "has_redistribution": has_redistribution,
        "n_tokens": n_tokens,
        "size": size,
    }


def classify_problem_type(problem_text: str) -> str:
    """
    题型分类：将网络流（max-flow / min-cost flow）从 transport_flow 拆出来，
    以便在 prompt 拼接时加入强约束护栏。
    """
    txt = (problem_text or "").lower()

    # network max-flow / min-cost flow
    is_network = any(w in txt for w in [
        "network", "node", "arc", "edge", "capacity", "capacities", "point", "station"
    ]) or any(w in txt for w in [
        "gb/s", "gigabytes per second", "data rate", "data center", "hub"
    ])

    asks_max = any(w in txt for w in [
        "maximum flow", "max flow", "maximum amount", "maximum data",
        "max amount", "max data", "maximize the amount", "how much can be sent",
        "what is the maximum amount", "maximum can be transferred",
        "find out the maximum", "the objective is to find out the maximum"
    ]) or ("maximize" in txt and "capacity" in txt)

    has_source_sink = any(w in txt for w in [
        "source", "sink", "from point", "to point", "from node", "to node",
        "from station", "to station", "from 0", "to 5"
    ])

    if is_network and asks_max and has_source_sink:
        return "network_max_flow"

    is_redistribution = any(w in txt for w in [
        "redistribute", "reallocate", "transfer", "ship", "transport", "move"
    ])
    has_costs = ("cost" in txt) or ("costs" in txt) or ("transportation cost" in txt)
    has_initial_required = any(w in txt for w in [
        "initial stock", "starts with", "starting stock", "required", "needs", "requirement"
    ])

    if is_network and has_costs and is_redistribution and has_initial_required and not asks_max:
        return "network_min_cost_flow"

    # selection / knapsack-like
    if any(w in txt for w in ["knapsack", "select", "choose"]) and any(w in txt for w in ["at most", "budget", "capacity", "limit"]):
        return "selection_knapsack"

    # assignment
    if ("assignment" in txt) or (("worker" in txt or "machine" in txt) and "task" in txt):
        return "assignment"

    # transportation / flow
    if any(w in txt for w in ["transport", "shipping", "shipment", "supply", "demand", "warehouse", "flow", "network", "transfer", "redistribute"]):
        return "transport_flow"

    # routing
    if any(w in txt for w in ["vehicle routing", "vrp", "route", "depot"]) and ("customer" in txt or "visit" in txt):
        return "routing_vrp"

    # production mix
    if any(w in txt for w in ["factory", "produce", "production", "manufacture"]) and any(w in txt for w in ["profit", "revenue", "cost", "capacity"]):
        return "production_mix"

    return "generic_lp_mip"


def _problem_guidance_compact(problem_type: str) -> List[str]:
    """
    为 system prompt 追加 guidance 块
    """
    pt = (problem_type or "").lower()

    if pt == "network_max_flow":
        return [
            "Type hint: MAX-FLOW in a directed capacitated network.",
            "- Use the standard max-flow template with a scalar total-flow variable F.",
            "- Objective MUST be maximize F (NEVER maximize a single arc flow[s][t]).",
            "- Flow conservation holds ONLY for intermediate nodes; source/sink use net flow == F.",
            "- Capacity constraints per arc: 0 <= flow[i][j] <= capacity[i][j] (only for existing arcs).",
            "- Missing arcs: either do not create variables for them OR enforce flow[i][j] == 0.",
            "- UNROLL constraints into explicit scalar constraints.",
        ]

    if pt == "network_min_cost_flow":
        return [
            "Type hint: MIN-COST redistribution / transshipment (min-cost flow).",
            "- Decision variable: shipped amount flow[i][j] (usually CONTINUOUS) on existing arcs.",
            "- Node balance: final[i] = initial[i] + inflow[i] - outflow[i].",
            "- If wording is 'needs/required/at least', enforce final[i] >= required[i] (NOT '==') unless totals match exactly.",
            "- Objective: minimize sum(cost[i][j]*flow[i][j]).",
            "- UNROLL constraints into explicit scalar constraints.",
        ]

    if pt == "selection_knapsack":
        return [
            "Type hint: selection/knapsack-like.",
            "- Use binary vars for select/not-select.",
            "- Encode 'must take' as x[item]=1, conflicts as x[a]+x[b]<=1, implications as x[a]<=x[b].",
        ]
    if pt == "assignment":
        return [
            "Type hint: assignment/matching.",
            "- Use binary vars x[w][t].",
            "- If constraints are 'for each task/worker', UNROLL them into explicit scalar constraints.",
        ]
    if pt == "transport_flow":
        return [
            "Type hint: transportation/flow.",
            "- Flow/shipment amounts are usually CONTINUOUS unless explicitly stated as integer units.",
            "- Balance constraints should be explicit per node (scalar, unrolled).",
        ]
    if pt == "routing_vrp":
        return [
            "Type hint: routing (VRP/VRPTW).",
            "- Vehicle count / visit decisions are typically INTEGER/BINARY.",
            "- If you cannot fully unroll a huge VRP, output a simplified but valid scalar-constraint model.",
        ]
    if pt == "production_mix":
        return [
            "Type hint: production mix.",
            "- Production quantity is INTEGER if it represents indivisible units; CONTINUOUS only if fraction is explicitly allowed.",
            "- Resource constraints: sum(usage*qty) <= capacity (but UNROLL per resource).",
        ]

    return [
        "Type hint: generic LP/MIP.",
        "- Identify sets/params/vars clearly and keep everything linear.",
    ]


def _vartype_guidance(feats: Dict[str, Any], problem_type: str) -> List[str]:
    """
    强化变量类型选择规则（continuous vs integer误判）
    """
    lines: List[str] = []
    lines.append("Variable type guidance (CRITICAL):")
    lines.append("- Default to INTEGER for decision variables that represent counts of indivisible things.")
    lines.append("- Use BINARY for yes/no decisions (select, assign, open/close, build/not build).")
    lines.append("- Use CONTINUOUS ONLY if the statement clearly allows fractional amounts OR the quantity is naturally divisible.")
    lines.append("  Common continuous signals: 'fractional', 'can be split', 'divisible', 'continuous', 'real-valued', 'any amount', 'half/0.5'.")
    lines.append("  Natural divisible quantities: flow/shipment amount, weight (kg/grams/tons), volume (liters), time (hours), energy, money, rates (GB/s).")

    if feats.get("is_diet_food"):
        if feats.get("allows_fraction"):
            lines.append("- Diet/meal plan: since fractional portions are allowed by the statement, servings/amounts may be CONTINUOUS.")
        else:
            lines.append("- Diet/meal plan: servings/portions/meals purchased are DISCRETE by default -> use INTEGER (unless fractional is explicitly allowed).")

    if problem_type in ("transport_flow", "network_max_flow", "network_min_cost_flow") or feats.get("has_flow"):
        if feats.get("allows_fraction") or feats.get("has_divisible_units"):
            lines.append("- Network/transport flow: flow variables are typically CONTINUOUS (unless stated as integer units/packets).")
        else:
            lines.append("- Network/transport flow: if the quantity is a rate/amount, prefer CONTINUOUS; if 'number of trucks/shipments', prefer INTEGER.")

    return lines


def _constraint_sense_guidance(feats: Dict[str, Any], problem_type: str) -> List[str]:
    """
    强化约束方向（<=, >=, ==）的选择
    """
    lines: List[str] = []
    lines.append("Constraint sense guidance (CRITICAL):")
    lines.append("- Map words to senses strictly:")
    lines.append("  * 'at least', 'no less than', 'needs', 'must have minimum', 'fulfill requirements' -> use '>='.")
    lines.append("  * 'at most', 'no more than', 'capacity', 'limit' -> use '<='.")
    lines.append("  * 'exactly', 'must equal', 'balanced exactly', 'equal to' -> use '=='.")
    lines.append("- DO NOT use '==' unless the statement explicitly requires exact equality.")

    if feats.get("has_initial_stock") and feats.get("has_required_need") and (feats.get("has_redistribution") or problem_type in ("transport_flow", "network_min_cost_flow")):
        lines.append("")
        lines.append("IMPORTANT for redistribution / rebalancing with initial vs required (avoid infeasibility):")
        lines.append("- Define final_stock[i] = initial_stock[i] + inflow[i] - outflow[i].")
        lines.append("- If wording is 'needs / required / fulfill / meet requirements', it means MINIMUM requirement:")
        lines.append("    final_stock[i] >= required_stock[i]    (NOT '==').")
        lines.append("- Only use 'final_stock[i] == required_stock[i]' if the statement says 'exactly' OR you FIRST verify totals:")
        lines.append("    sum(initial_stock) == sum(required_stock).")
        lines.append("- If totals do NOT match and you still want equalities, add surplus/waste variables, e.g.:")
        lines.append("    final_stock[i] == required_stock[i] + surplus[i], surplus[i] >= 0.")

    return lines


def _maxflow_guidance_by_ptype(problem_type: str) -> List[str]:
    """
    MAX-FLOW 专用
    """
    if (problem_type or "").lower() != "network_max_flow":
        return []
    return [
        "MAX-FLOW NETWORK MODELING GUARDRAIL (CRITICAL):",
        "- Model is MAX-FLOW, NOT a circulation.",
        "- DO NOT enforce inflow==outflow at SOURCE or SINK.",
        "- Introduce a scalar variable F >= 0 for total flow from source to sink.",
        "- Objective MUST be maximize F (NEVER maximize a single arc flow[source][sink]).",
        "- Source constraint:  sum_out(flow[source][*]) - sum_in(flow[*][source]) == F.",
        "- Sink constraint:    sum_in(flow[*][sink]) - sum_out(flow[sink][*]) == F.",
        "- For intermediate node k:  sum_in(flow[*][k]) == sum_out(flow[k][*]).",
        "- Capacity per arc: 0 <= flow[i][j] <= capacity[i][j] for existing arcs.",
        "- Missing arcs: either do NOT create variables for them OR enforce flow[i][j] == 0.",
        "- UNROLL into explicit scalar constraints (no comprehensions in constraints).",
    ]


def build_system_prompt(
    question_text: str,
    enable_dynamic: bool = True,
    difficulty_hint: Optional[str] = None,
) -> str:
    """
    Dynamic system prompt templating:
    - 保留 BASE_SYSTEM_PROMPT
    - 根据题面特征与题型追加 guidance 块
    """
    if not enable_dynamic:
        return BASE_SYSTEM_PROMPT

    feats = _simple_text_features(question_text)
    ptype = classify_problem_type(question_text)

    type_guidance = _problem_guidance_compact(ptype)
    vartype_guidance = _vartype_guidance(feats, ptype)
    sense_guidance = _constraint_sense_guidance(feats, ptype)
    maxflow_guidance = _maxflow_guidance_by_ptype(ptype)

    lines: List[str] = [BASE_SYSTEM_PROMPT, ""]

    lines.append("INTERNAL GUIDANCE (do NOT include this text in the JSON output):")
    lines.append("- Output JSON ONLY. No markdown, no explanation text.")
    lines.append("- Constraints MUST be fully specified SCALAR constraints.")
    lines.append("- In constraints, NEVER use generator expressions or comprehensions.")
    lines.append("- If the statement says 'for each/for all', you MUST UNROLL into one constraint per element.")
    lines.append("- Use string indices like x['I']['A'] (not x[I][A], not x[1]).")

    # Feature-triggered hints
    if feats.get("has_table"):
        lines.append("- The statement likely contains a table: convert rows/cols into SET elements and PARAM values explicitly.")
    if feats.get("has_for_each"):
        lines.append("- The statement has quantifiers ('for each/for all'): UNROLL them into explicit scalar constraints.")
    if feats.get("has_capacity"):
        lines.append("- Capacity/limit constraints: write explicit per-resource/per-period scalar constraints.")
    if feats.get("has_implication"):
        lines.append("- Logical rules (if/then/must/cannot): encode using linear inequalities (x<=y, x+y<=1, etc.).")
    if feats.get("has_assignment"):
        lines.append("- Assignment-like structure: binary vars with unrolled one-to-one constraints.")
    if feats.get("has_flow"):
        lines.append("- Flow-like structure: balance constraints should be unrolled per node.")
    if feats.get("has_routing"):
        lines.append("- Routing is complex: prefer a smaller valid model rather than implicit quantifiers.")
    if feats.get("has_production"):
        lines.append("- Production mix: check units, capacities, and profit definition carefully.")

    lines.append("")
    lines.extend(vartype_guidance)

    lines.append("")
    lines.extend(sense_guidance)

    if maxflow_guidance:
        lines.append("")
        lines.extend(maxflow_guidance)

    lines.append("")
    lines.extend(type_guidance)

    if difficulty_hint:
        lines.append("")
        lines.append(f"Difficulty hint: {difficulty_hint}. Be extra systematic with indices and unrolling.")

    return "\n".join(lines).strip()


def build_user_prompt(question_text: str) -> str:
    """
    Build the user message content: include schema/instructions + the NL question.
    """
    return (
        SCHEMA_AND_INSTRUCTIONS
        + "\n\nNow read the following optimization problem and output the JSON IR:\n\n"
        + question_text
    )


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from a model output text.

    Supports two common patterns:
    - A fenced code block ```json ... ```
    - A raw JSON object in the text
    """
    code_block_pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
    m = code_block_pattern.search(text)
    if m:
        json_str = m.group(1)
        return json.loads(json_str)

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        json_str = text[first : last + 1]
        return json.loads(json_str)

    return json.loads(text)


def _filter_kwargs_for(cls, raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter raw dict to only include fields that exist in dataclass `cls`.
    Drop extra keys (e.g., LLM sometimes adds unknown keys).
    """
    field_names = {f.name for f in fields(cls)}
    extra_keys = [k for k in raw.keys() if k not in field_names]
    if extra_keys:
        print(f"[NL2IR WARN] Dropping unknown keys for {cls.__name__}: {extra_keys}")
    return {k: v for k, v in raw.items() if k in field_names}


def json_to_model_ir(data: Dict[str, Any]) -> ModelIR:
    """
    Convert JSON dict (following the schema) to a ModelIR instance.
    Unknown / extra keys in objects will be dropped.
    """
    meta = MetaInfo(**_filter_kwargs_for(MetaInfo, data["meta"]))

    sets = [
        SetDef(**_filter_kwargs_for(SetDef, s))
        for s in data.get("sets", [])
    ]

    params = [
        ParamDef(**_filter_kwargs_for(ParamDef, p))
        for p in data.get("params", [])
    ]

    vars_ = [
        VarDef(**_filter_kwargs_for(VarDef, v))
        for v in data.get("vars", [])
    ]

    obj = ObjectiveDef(**_filter_kwargs_for(ObjectiveDef, data["objective"]))

    constraints = [
        ConstraintDef(**_filter_kwargs_for(ConstraintDef, c))
        for c in data.get("constraints", [])
    ]

    ir = ModelIR(
        meta=meta,
        sets=sets,
        params=params,
        vars=vars_,
        objective=obj,
        constraints=constraints,
        graph=None,
    )
    return ir
