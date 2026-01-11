# ir2solve_nl2ir.py
# NL -> JSON IR text -> ModelIR parsing utilities
#   - NL2IR prompt provides a *compile-safe baseline* (strict JSON, scalar constraints,
#     canonical indexing, conservative integrality).

from __future__ import annotations

import json
import re
from dataclasses import fields
from typing import Any, Dict, Optional

from ir2solve_ir import (
    MetaInfo,
    SetDef,
    ParamDef,
    VarDef,
    ObjectiveDef,
    ConstraintDef,
    ModelIR,
)

# =============================================================================
# 1) Prompt Constants (compile-safe baseline)
# =============================================================================

BASE_SYSTEM_PROMPT = """
You are an expert in mathematical modeling and optimization.

Read a natural-language description of a linear / integer optimization problem and
output a JSON object describing the model in a strictly structured IR called ModelIR.

Hard requirements:
- Output JSON ONLY (no markdown, no explanations).
- Produce valid JSON (double quotes only; no trailing commas; no comments).
- Follow the schema exactly. Do NOT add extra keys.
- Use Python expressions in strings for objective/constraints (only arithmetic, indexing, quicksum/sum).
""".strip()

SCHEMA_AND_INSTRUCTIONS = r"""
Generate ONE JSON object with these top-level keys:
- meta
- sets
- params
- vars
- objective
- constraints

==== Schema ====

1) meta
{
  "problem_id": "string or null",
  "source": "string or null",
  "description": "string or null",
  "sense": "min" or "max",
  "version": "string or null"
}

2) sets
A list of sets. Each set is:
{
  "name": "string",
  "elements": ["string", ...],   // IMPORTANT: every element MUST be a STRING (even if it looks like a number)
  "description": "string or null"
}

3) params
A list of parameters. Each param is:
{
  "name": "string",
  "indices": [] | ["SetName"] | ["SetName1","SetName2"],  // 0D/1D/2D only
  "values": number | {key:number,...} | {i:{j:number,...},...},   // recommended canonical forms below
  "description": "string or null"
}

Canonical forms for values:
- Scalar (0D): a number (or a single-entry dict is tolerated by the compiler)
- 1D over set I: {"i1": 1.0, "i2": 2.0, ...}
- 2D over sets I,J (RECOMMENDED): {"i1": {"j1": 1.0, "j2": 3.0}, "i2": {...}, ...}

4) vars
A list of decision variables. Each var is:
{
  "name": "string",
  "indices": [] | ["SetName"] | ["SetName1","SetName2"],  // 0D/1D/2D only
  "vartype": "continuous" | "integer" | "binary",
  "lb": number,
  "ub": number or null,
  "description": "string or null"
}

IMPORTANT (integrality baseline):
- Default to "integer" for counts/units/number of items/visits/vehicles/facilities/assignments.
- Use "binary" only for yes/no decisions.
- Use "continuous" only for divisible amounts (flow, blending amount, time, money, proportion, continuous production).

5) objective
{
  "name": "string",
  "sense": "min" or "max",
  "expr": "Python expression string",
  "description": "string or null"
}

6) constraints
A list of constraints. Each constraint is:
{
  "name": "string",
  "expr_lhs": "Python expression string",
  "sense": "<=" | ">=" | "==",
  "expr_rhs": "Python expression string",
  "description": "string or null"
}

==== Expression rules (compile-safe baseline) ====

Allowed building blocks inside expr strings:
- Numbers, + - * /, parentheses
- Indexing into params/vars: x[i], x[i][j], cost[i][j], demand[i]
- Aggregation: quicksum( ... for i in I ), sum( ... for i in I )
  (You may nest sums, e.g., quicksum(quicksum(x[i][j] for j in J) for i in I))

Hard constraints (must follow):
- Every constraint must be a SINGLE SCALAR constraint (lhs and rhs each evaluate to a scalar).
- Do NOT write implicit "for all" constraints inside one constraint (e.g., "for i in I: ...").
  If a constraint must hold for each i, EXPAND it into multiple entries in the constraints list.
- Do NOT use free index symbols (i/j/t/...) outside of a sum/generator that binds them.
- Do NOT use list comprehensions that build lists; only generator expressions inside sum/quicksum.

==== Output template (fill it; keep keys exactly) ====

{
  "meta": {
    "problem_id": null,
    "source": null,
    "description": null,
    "sense": "min",
    "version": "v1"
  },
  "sets": [
    {
      "name": "I",
      "elements": ["A", "B"],
      "description": null
    }
  ],
  "params": [
    {
      "name": "cost",
      "indices": ["I"],
      "values": {"A": 1.0, "B": 2.0},
      "description": null
    }
  ],
  "vars": [
    {
      "name": "x",
      "indices": ["I"],
      "vartype": "integer",
      "lb": 0.0,
      "ub": null,
      "description": null
    }
  ],
  "objective": {
    "name": "obj",
    "sense": "min",
    "expr": "quicksum(cost[i] * x[i] for i in I)",
    "description": null
  },
  "constraints": [
    {
      "name": "c1",
      "expr_lhs": "x['A'] + x['B']",
      "sense": ">=",
      "expr_rhs": "1.0",
      "description": null
    }
  ]
}

Before outputting, self-check:
- JSON parses; keys match schema; no extra keys.
- All referenced names (sets/params/vars) are defined.
- No free indices; per-index constraints are expanded.
""".strip()


def build_system_prompt() -> str:
    return BASE_SYSTEM_PROMPT


def build_user_prompt(question_text: str) -> str:
    return (
        SCHEMA_AND_INSTRUCTIONS
        + "\n\nNow read the following optimization problem and output the JSON IR (JSON ONLY):\n\n"
        + (question_text or "")
    )


# =============================================================================
# 2) Output Extraction / Parsing
# =============================================================================

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _find_first_balanced_json_object(raw: str) -> Optional[str]:
    """
    Return the first balanced {...} JSON object substring if found.
    This is a robust fallback when the model outputs surrounding text.
    """
    if not raw:
        return None
    start = raw.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start : idx + 1]
    return None


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from model output text.

    Supports:
    - A fenced code block ```json ... ``` (also tolerates ``` ... ```)
    - A raw JSON object embedded in text (balanced brace scan)
    - A raw JSON string (already JSON)
    """
    raw = text or ""

    m = _CODE_BLOCK_RE.search(raw)
    if m:
        return json.loads(m.group(1))

    obj = _find_first_balanced_json_object(raw)
    if obj is not None:
        return json.loads(obj)

    return json.loads(raw)


# =============================================================================
# 3) JSON -> ModelIR
# =============================================================================

def _filter_kwargs_for(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only fields that exist on the dataclass `cls`.
    This makes parsing robust to minor extra keys, while still encouraging schema adherence.
    """
    if not isinstance(data, dict):
        return {}
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


def json_to_model_ir(data: Dict[str, Any]) -> ModelIR:
    """
    Convert a parsed JSON dict into ModelIR dataclasses.
    Unknown fields are ignored (minimal robustness); schema violations should be caught by verifier.
    """
    if not isinstance(data, dict):
        raise TypeError("ModelIR JSON must be an object (dict).")

    meta = MetaInfo(**_filter_kwargs_for(MetaInfo, data.get("meta") or {}))

    sets = [SetDef(**_filter_kwargs_for(SetDef, s)) for s in (data.get("sets") or [])]
    params = [ParamDef(**_filter_kwargs_for(ParamDef, p)) for p in (data.get("params") or [])]
    vars_ = [VarDef(**_filter_kwargs_for(VarDef, v)) for v in (data.get("vars") or [])]

    obj_raw = data.get("objective") or {}
    obj = ObjectiveDef(**_filter_kwargs_for(ObjectiveDef, obj_raw))

    constraints = [
        ConstraintDef(**_filter_kwargs_for(ConstraintDef, c))
        for c in (data.get("constraints") or [])
    ]

    return ModelIR(
        meta=meta,
        sets=sets,
        params=params,
        vars=vars_,
        objective=obj,
        constraints=constraints,
    )
