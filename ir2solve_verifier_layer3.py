# verifier_layer3.py
# Layer 3: Type-specific template rescue
#
# Workflow:
# 1) Type identification (high-confidence score):
#    - Text keywords (meta.description if available) +
#      IR fingerprints (sets/params/vars names & shapes).
# 2) Full rebuild via LLM using:
#    - Base NL2IR prompts (ir2solve_nl2ir.py) +
#    - Type-specific instruction block (this file).
# 3) Acceptance test:
#    - Build succeeds (ir_to_gurobi) AND solver returns FEASIBLE/OPTIMAL (short time limit).
#
# Types supported:
# - max_flow (network flow / maximum flow)
# - assignment
# - knapsack

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re

from ir2solve_verifier_core import VerifierRule, RuleDetection, mk_issue, mk_repair


# -----------------------------------------------------------------------------
# Type identification
# -----------------------------------------------------------------------------

def _lc(x: Optional[str]) -> str:
    return (x or "").strip().lower()


def _has_any(text: str, kws: List[str]) -> bool:
    t = text.lower()
    return any(k in t for k in kws)


def _score_keywords(text: str, kws: List[str]) -> float:
    t = text.lower()
    hits = sum(1 for k in kws if k in t)
    if not kws:
        return 0.0
    return min(1.0, hits / max(2, len(kws) * 0.5))


def _set_names(ir: Any) -> List[str]:
    return [str(getattr(s, "name", "")) for s in getattr(ir, "sets", []) or [] if getattr(s, "name", None)]


def _param_summaries(ir: Any) -> List[Tuple[str, List[str]]]:
    out = []
    for p in getattr(ir, "params", []) or []:
        out.append((str(getattr(p, "name", "")), list(getattr(p, "indices", []) or [])))
    return out


def _var_summaries(ir: Any) -> List[Tuple[str, List[str], str]]:
    out = []
    for v in getattr(ir, "vars", []) or []:
        out.append((str(getattr(v, "name", "")), list(getattr(v, "indices", []) or []), str(getattr(v, "vartype", ""))))
    return out


def _ir_fingerprint_score(ir: Any, kind: str) -> float:
    sets = [s.lower() for s in _set_names(ir)]
    params = [(n.lower(), [x.lower() for x in idx]) for (n, idx) in _param_summaries(ir)]
    vars_ = [(n.lower(), [x.lower() for x in idx], vt.lower()) for (n, idx, vt) in _var_summaries(ir)]

    def has_set_like(names: List[str]) -> bool:
        return any(any(k in s for k in names) for s in sets)

    def has_param_like(names: List[str], min_dim: int = 0) -> bool:
        for n, idx in params:
            if any(k in n for k in names) and len(idx) >= min_dim:
                return True
        return False

    def has_var_like(names: List[str], min_dim: int = 0) -> bool:
        for n, idx, _vt in vars_:
            if any(k in n for k in names) and len(idx) >= min_dim:
                return True
        return False

    score = 0.0

    if kind == "max_flow":
        # Nodes/Edges + capacity + flow are strong signals
        if has_set_like(["node", "nodes", "v", "n"]):
            score += 0.25
        if has_set_like(["edge", "edges", "arc", "arcs", "a", "e"]):
            score += 0.25
        if has_param_like(["cap", "capacity"], min_dim=1):
            score += 0.25
        if has_var_like(["flow", "f"], min_dim=1):
            score += 0.25
        return min(1.0, score)

    if kind == "assignment":
        # Workers/Tasks + cost matrix + binary assignment var
        if has_set_like(["worker", "workers"]):
            score += 0.25
        if has_set_like(["task", "tasks", "job", "jobs"]):
            score += 0.25
        if has_param_like(["cost", "c"], min_dim=2):
            score += 0.25
        # var x/assign with 2D, often binary
        for n, idx, vt in vars_:
            if len(idx) == 2 and (("assign" in n) or (n in ("x", "y"))):
                score += 0.15
                if vt == "binary":
                    score += 0.10
                break
        return min(1.0, score)

    if kind == "knapsack":
        if has_set_like(["item", "items"]):
            score += 0.30
        if has_param_like(["weight", "w"], min_dim=1):
            score += 0.25
        if has_param_like(["value", "profit", "benefit", "v"], min_dim=1):
            score += 0.25
        if has_param_like(["capacity", "cap", "limit"], min_dim=0):
            score += 0.10
        # decision var x over items
        for n, idx, vt in vars_:
            if len(idx) == 1 and (n in ("x", "take", "select") or "select" in n or "take" in n):
                score += 0.10
                break
        return min(1.0, score)

    return 0.0


def identify_type(ir: Any) -> Tuple[Optional[str], float, Dict[str, float]]:
    """
    Return (best_kind, best_score, per_kind_scores).
    We combine:
      score = 0.55 * keyword_score + 0.45 * ir_fingerprint_score
    Text source: ir.meta.description (if present) else empty.
    """
    meta = getattr(ir, "meta", None)
    text = _lc(getattr(meta, "description", "")) if meta is not None else ""

    # keyword sets
    kw = {
        "max_flow": ["max flow", "maximum flow", "flow", "source", "sink", "capacity", "arc", "edge", "node"],
        "assignment": ["assign", "assignment", "worker", "task", "job", "exactly one", "at most one", "matching"],
        "knapsack": ["knapsack", "capacity", "weight", "value", "profit", "choose", "select", "items"],
    }

    scores: Dict[str, float] = {}
    for k in ("max_flow", "assignment", "knapsack"):
        s_kw = _score_keywords(text, kw[k]) if text else 0.0
        s_ir = _ir_fingerprint_score(ir, k)
        scores[k] = 0.55 * s_kw + 0.45 * s_ir

    best_kind = max(scores, key=lambda k: scores[k])
    best_score = scores[best_kind]
    return best_kind, best_score, scores


# -----------------------------------------------------------------------------
# LLM prompts (type-specific add-ons)
# -----------------------------------------------------------------------------

TYPE_PROMPTS: Dict[str, str] = {
    "max_flow": r"""
You MUST model this problem as a single-source single-sink MAX-FLOW problem (NOT a circulation).

Model requirements:
- Introduce a scalar variable F >= 0 representing total flow from source to sink.
    Objective MUST be: maximize F. NEVER maximize a single arc flow[source][sink].
- Do NOT enforce inflow==outflow at SOURCE or SINK.
    Source balance: sum_out(flow[source,*]) - sum_in(flow[* ,source]) == F
    Sink balance:   sum_in(flow[* ,sink]) - sum_out(flow[sink,*]) == F
- For every intermediate node k != source,sink:
    sum_in(flow[* ,k]) == sum_out(flow[k,*])
- Capacity on existing arcs only:
    For each existing arc (i,j): 0 <= flow[i][j] <= capacity[i][j]
    Missing arcs: either do NOT create variables for them OR enforce flow[i][j] == 0.
- UNROLL constraints into explicit scalar constraints (no implicit "for all").
""".strip(),

    "assignment": r"""
You MUST model this problem as a standard ASSIGNMENT / MATCHING problem.

Model requirements:
- Sets: Workers W, Tasks T.
- Costs: cost[w][t].
- Decision variables: x[w][t] ∈ {0,1}.
- Each task assigned exactly once:
    for each t in T: sum_{w in W} x[w][t] == 1
- Each worker used at most once (or exactly once if stated):
    for each w in W: sum_{t in T} x[w][t] <= 1
- Objective: minimize total cost sum_{w,t} cost[w][t] * x[w][t]
""".strip(),

    "knapsack": r"""
You MUST model this problem as a 0-1 KNAPSACK problem.

Model requirements:
- Set: Items I.
- Params: weight[i], value[i] (or profit[i]); capacity C.
- Decision variables: x[i] ∈ {0,1}.
- Capacity constraint: sum_i weight[i] * x[i] <= C.
- Objective: maximize sum_i value[i] * x[i].
""".strip(),
}


# -----------------------------------------------------------------------------
# Fallback "question text" construction
# -----------------------------------------------------------------------------

def _fallback_problem_text_from_ir(ir: Any, max_chars: int = 4000) -> str:
    """
    If meta.description is missing, create a compact textual summary from IR.
    This is only a fallback for L3 rebuild prompts.
    """
    parts: List[str] = []
    meta = getattr(ir, "meta", None)
    if meta is not None:
        parts.append(f"Problem meta.sense: {getattr(meta,'sense',None)}")
        if getattr(meta, "source", None):
            parts.append(f"Source: {getattr(meta,'source',None)}")

    # sets
    parts.append("Sets:")
    for s in getattr(ir, "sets", []) or []:
        name = getattr(s, "name", "")
        elems = getattr(s, "elements", []) or []
        # truncate
        sample = elems[:10]
        parts.append(f"- {name}: {sample} {'...' if len(elems)>10 else ''}")

    # params names and indices
    parts.append("Params:")
    for p in getattr(ir, "params", []) or []:
        parts.append(f"- {getattr(p,'name','')}: indices={getattr(p,'indices',[])}")

    # vars names and indices
    parts.append("Vars:")
    for v in getattr(ir, "vars", []) or []:
        parts.append(f"- {getattr(v,'name','')}: indices={getattr(v,'indices',[])} vartype={getattr(v,'vartype','')}")

    # objective / a few constraints
    obj = getattr(ir, "objective", None)
    if obj is not None:
        parts.append(f"Objective: sense={getattr(obj,'sense',None)} expr={getattr(obj,'expr','')}")
    parts.append("Constraints (sample):")
    for c in (getattr(ir, "constraints", []) or [])[:10]:
        parts.append(f"- {getattr(c,'name','')}: {getattr(c,'expr_lhs','')} {getattr(c,'sense','')} {getattr(c,'expr_rhs','')}")

    out = "\n".join(parts)
    return out[:max_chars]


# -----------------------------------------------------------------------------
# Acceptance test
# -----------------------------------------------------------------------------

def acceptance_test(ir: Any, time_limit: float = 5.0) -> Tuple[bool, str]:
    """
    Build + optimize quickly; accept if status indicates feasible/optimal.
    Returns (ok, status_name).
    """
    try:
        from ir2solve_ir import ir_to_gurobi
        from gurobipy import GRB
    except Exception as e:
        return False, f"tooling_import_error:{type(e).__name__}"

    try:
        m = ir_to_gurobi(ir)
        try:
            m.Params.OutputFlag = 0
        except Exception:
            pass
        try:
            m.Params.TimeLimit = float(time_limit)
        except Exception:
            pass
        m.optimize()

        st = m.Status
        # Accept OPTIMAL; also accept FEASIBLE if stopped early.
        if st == GRB.OPTIMAL:
            return True, "OPTIMAL"
        if st in (GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            # Check feasibility
            try:
                if m.SolCount and m.SolCount > 0:
                    return True, "FEASIBLE"
            except Exception:
                pass
        return False, str(st)
    except Exception as e:
        return False, f"build_or_opt_error:{type(e).__name__}"


# -----------------------------------------------------------------------------
# Main rule
# -----------------------------------------------------------------------------

class TypeTemplateRescue(VerifierRule):
    layer = "L3"
    kind = "type_template_rescue"

    # High-confidence threshold
    THRESHOLD = 0.75

    def detect(self, ir: Any) -> Optional[RuleDetection]:
        kind, score, scores = identify_type(ir)
        if kind is None:
            return None
        if score < self.THRESHOLD:
            return None

        return RuleDetection(
            issue=mk_issue(
                layer=self.layer,
                kind=self.kind,
                severity="info",
                message=f"High-confidence type identified: {kind} (score={score:.2f}). Attempting L3 LLM rebuild + acceptance test.",
                nodes=[kind],
            ),
            data={"kind": kind, "score": score, "scores": scores},
        )

    def apply(self, ir: Any, detection: RuleDetection) -> Optional[Dict[str, Any]]:
        kind = detection.data.get("kind")
        if kind not in TYPE_PROMPTS:
            return None

        # Prepare "problem text"
        meta = getattr(ir, "meta", None)
        base_text = _lc(getattr(meta, "description", "")) if meta is not None else ""
        if not base_text:
            base_text = _fallback_problem_text_from_ir(ir)

        # Build prompts: base NL2IR prompts + type-specific add-on
        try:
            from ir2solve_nl2ir import build_system_prompt, build_user_prompt, extract_json_from_text, json_to_model_ir
        except Exception as e:
            return mk_repair(
                layer=self.layer,
                kind=self.kind,
                message=f"Cannot import ir2solve_nl2ir helpers for L3 rebuild: {type(e).__name__}",
                changed_fields=[],
            )

        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(base_text) + "\n\n" + "TYPE-SPECIFIC INSTRUCTIONS:\n" + TYPE_PROMPTS[kind]

        # Call LLM
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            # Report-only: do not modify IR
            return None

        # Parse new IR
        try:
            data = extract_json_from_text(raw)
            new_ir = json_to_model_ir(data)
        except Exception:
            return None

        # Acceptance test
        ok, st = acceptance_test(new_ir, time_limit=5.0)
        if not ok:
            return None

        # Apply: replace the IR fields in-place
        setattr(ir, "meta", getattr(new_ir, "meta", getattr(ir, "meta", None)))
        setattr(ir, "sets", getattr(new_ir, "sets", getattr(ir, "sets", [])))
        setattr(ir, "params", getattr(new_ir, "params", getattr(ir, "params", [])))
        setattr(ir, "vars", getattr(new_ir, "vars", getattr(ir, "vars", [])))
        setattr(ir, "objective", getattr(new_ir, "objective", getattr(ir, "objective", None)))
        setattr(ir, "constraints", getattr(new_ir, "constraints", getattr(ir, "constraints", [])))

        return mk_repair(
            layer=self.layer,
            kind=self.kind,
            message=f"L3 rebuilt model as {kind} and passed acceptance test ({st}).",
            changed_fields=["meta", "sets", "params", "vars", "objective", "constraints"],
        )


def get_layer3_rules() -> List[VerifierRule]:
    return [TypeTemplateRescue()]
