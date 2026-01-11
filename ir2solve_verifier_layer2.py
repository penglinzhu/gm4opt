# verifier_layer2.py
# Layer 2: Generic semantic checks & repairs
#
# Scope:
# 1) Integrality sanity:
#    - In absence of clear evidence, avoid continuous variables for clearly discrete decisions.
# 2) Constraint direction sanity:
#    - Avoid confusing "==" with "at least / at most" when the constraint name/description indicates it.
# 3) Generic structure-semantic normalization:
#    - Align objective sense with meta.sense; normalize constraint sense tokens;
#      enforce binary bounds (lb=0, ub=1) for binary vars.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re

from ir2solve_verifier_core import (
    VerifierRule,
    RuleDetection,
    mk_issue,
    mk_repair,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _lc(x: Optional[str]) -> str:
    return (x or "").strip().lower()


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


# Conservative hints for "discrete decision" (safe-ish to upgrade continuous -> integer/binary)
_DISCRETE_VAR_HINTS = [
    "assign", "assignment",
    "select", "selection",
    "choose", "pick", "take",
    "open", "close",
    "build", "install",
    "use", "activate",
    "visit", "route",
    "serve",
    "facility", "facilities",
    "worker", "workers",
    "task", "tasks",
    "item", "items",
    "node", "nodes",
    "edge", "edges",
    "arc", "arcs",
]

# Hints for sets indicating "combinatorial/discrete index"
_DISCRETE_SET_HINTS = [
    "worker", "workers",
    "task", "tasks",
    "item", "items",
    "facility", "facilities",
    "node", "nodes",
    "edge", "edges",
    "arc", "arcs",
    "city", "cities",
    "job", "jobs",
]

# For direction sanity: keywords implying inequality meaning
_AT_LEAST_KWS = ["at least", "no less", "minimum", "min ", ">= ", "atleast"]
_AT_MOST_KWS  = ["at most", "no more", "maximum", "max ", "<= ", "atmost", "up to", "upto", "limit"]

# A tiny sense normalizer
_SENSE_MAP = {
    "=": "==",
    "==": "==",
    "<=": "<=",
    ">=": ">=",
    "≤": "<=",
    "≥": ">=",
}


def _iter_vars(ir: Any):
    return list(getattr(ir, "vars", []) or [])


def _iter_constraints(ir: Any):
    return list(getattr(ir, "constraints", []) or [])


def _set_name_hints(ir: Any, set_name: str) -> str:
    # use set name + (optional) description as hint
    for s in getattr(ir, "sets", []) or []:
        if getattr(s, "name", None) == set_name:
            return f"{set_name} {getattr(s, 'description', '') or ''}"
    return set_name


# -----------------------------------------------------------------------------
# Rule 1: Integrality sanity (conservative upgrades)
# -----------------------------------------------------------------------------

class IntegralitySanity(VerifierRule):
    layer = "L2"
    kind = "integrality_sanity"

    def detect(self, ir: Any) -> Optional[RuleDetection]:
        offenders: List[str] = []
        for v in _iter_vars(ir):
            vt = _lc(getattr(v, "vartype", ""))
            if vt != "continuous":
                continue

            name = _lc(getattr(v, "name", ""))
            desc = _lc(getattr(v, "description", ""))
            idx = getattr(v, "indices", []) or []
            idx_hint = " ".join(_lc(_set_name_hints(ir, s)) for s in idx if isinstance(s, str))

            # Discrete evidence: variable name/desc or index set hint
            discrete_evidence = (
                _contains_any(name, _DISCRETE_VAR_HINTS)
                or _contains_any(desc, _DISCRETE_VAR_HINTS)
                or _contains_any(idx_hint, _DISCRETE_SET_HINTS)
            )

            if discrete_evidence:
                offenders.append(getattr(v, "name", ""))
        if not offenders:
            return None

        return RuleDetection(
            issue=mk_issue(
                layer=self.layer,
                kind=self.kind,
                severity="warning",
                message="Some continuous variables look like discrete decisions; consider upgrading vartype to integer/binary.",
                nodes=offenders,
            ),
            data={"offenders": offenders},
        )

    def apply(self, ir: Any, detection: RuleDetection) -> Optional[Dict[str, Any]]:
        offenders = set(detection.data.get("offenders", []) or [])
        changed_fields: List[str] = []

        for v in _iter_vars(ir):
            if getattr(v, "name", "") not in offenders:
                continue

            lb = getattr(v, "lb", None)
            ub = getattr(v, "ub", None)

            # If bounds already indicate binary, make it binary; else integer.
            if lb == 0 or lb == 0.0:
                if ub == 1 or ub == 1.0:
                    new_type = "binary"
                else:
                    new_type = "integer"
            else:
                new_type = "integer"

            if _lc(getattr(v, "vartype", "")) != new_type:
                setattr(v, "vartype", new_type)
                changed_fields.append(f"vars[{getattr(v,'name','?')}].vartype")

            # Keep binary bounds consistent (safe normalization)
            if new_type == "binary":
                if lb not in (0, 0.0):
                    setattr(v, "lb", 0.0)
                    changed_fields.append(f"vars[{getattr(v,'name','?')}].lb")
                if ub not in (1, 1.0):
                    setattr(v, "ub", 1.0)
                    changed_fields.append(f"vars[{getattr(v,'name','?')}].ub")

        if not changed_fields:
            return None

        return mk_repair(
            layer=self.layer,
            kind=self.kind,
            message="Upgraded clearly-discrete continuous variables to integer/binary (conservative).",
            changed_fields=changed_fields,
        )


# -----------------------------------------------------------------------------
# Rule 2: Constraint direction sanity (== vs at least/at most)
# -----------------------------------------------------------------------------

class ConstraintDirectionSanity(VerifierRule):
    layer = "L2"
    kind = "constraint_direction_sanity"

    def detect(self, ir: Any) -> Optional[RuleDetection]:
        offenders: List[str] = []
        for c in _iter_constraints(ir):
            sense = _SENSE_MAP.get(_lc(getattr(c, "sense", "")), getattr(c, "sense", ""))
            if sense != "==":
                continue

            name = _lc(getattr(c, "name", ""))
            desc = _lc(getattr(c, "description", ""))
            hint = f"{name} {desc}"

            if _contains_any(hint, _AT_LEAST_KWS) or _contains_any(hint, _AT_MOST_KWS):
                offenders.append(getattr(c, "name", ""))
        if not offenders:
            return None

        return RuleDetection(
            issue=mk_issue(
                layer=self.layer,
                kind=self.kind,
                severity="warning",
                message="Some constraints use '==' but appear to mean 'at least' or 'at most' (from name/description).",
                nodes=offenders,
            ),
            data={"offenders": offenders},
        )

    def apply(self, ir: Any, detection: RuleDetection) -> Optional[Dict[str, Any]]:
        offenders = set(detection.data.get("offenders", []) or [])
        changed_fields: List[str] = []

        for c in _iter_constraints(ir):
            if getattr(c, "name", "") not in offenders:
                continue

            name = _lc(getattr(c, "name", ""))
            desc = _lc(getattr(c, "description", ""))
            hint = f"{name} {desc}"

            # Determine intended direction. If both cues appear, do nothing (ambiguous).
            want_ge = _contains_any(hint, _AT_LEAST_KWS)
            want_le = _contains_any(hint, _AT_MOST_KWS)
            if want_ge and want_le:
                continue

            if want_ge and _lc(getattr(c, "sense", "")) != ">=":
                setattr(c, "sense", ">=")
                changed_fields.append(f"constraints[{getattr(c,'name','?')}].sense")
            elif want_le and _lc(getattr(c, "sense", "")) != "<=":
                setattr(c, "sense", "<=")
                changed_fields.append(f"constraints[{getattr(c,'name','?')}].sense")

        if not changed_fields:
            return None

        return mk_repair(
            layer=self.layer,
            kind=self.kind,
            message="Adjusted constraint senses based on at-least/at-most cues in constraint name/description.",
            changed_fields=changed_fields,
        )


# -----------------------------------------------------------------------------
# Rule 3: Generic structure-semantic normalization
# -----------------------------------------------------------------------------

class GenericStructureSemanticSanity(VerifierRule):
    layer = "L2"
    kind = "generic_structure_semantic_sanity"

    def detect(self, ir: Any) -> Optional[RuleDetection]:
        nodes: List[str] = []
        data: Dict[str, Any] = {}

        meta = getattr(ir, "meta", None)
        obj = getattr(ir, "objective", None)

        # Objective sense mismatch with meta.sense
        ms = _lc(getattr(meta, "sense", "")) if meta is not None else ""
        os = _lc(getattr(obj, "sense", "")) if obj is not None else ""
        if ms in ("min", "max") and os in ("min", "max") and ms != os:
            nodes.append("objective.sense")
            data["objective_sense_mismatch"] = (ms, os)

        # Constraint sense token normalization needed
        bad_sense: List[str] = []
        for c in _iter_constraints(ir):
            s = getattr(c, "sense", "")
            if s not in ("<=", ">=", "==") and _lc(s) in _SENSE_MAP:
                bad_sense.append(getattr(c, "name", ""))
        if bad_sense:
            nodes.append("constraints.sense")
            data["bad_constraint_sense"] = bad_sense

        # Binary bounds normalization needed
        bad_bounds: List[str] = []
        for v in _iter_vars(ir):
            if _lc(getattr(v, "vartype", "")) != "binary":
                continue
            lb = getattr(v, "lb", None)
            ub = getattr(v, "ub", None)
            if lb not in (0, 0.0) or ub not in (1, 1.0):
                bad_bounds.append(getattr(v, "name", ""))
        if bad_bounds:
            nodes.append("vars.binary_bounds")
            data["bad_binary_bounds"] = bad_bounds

        if not nodes:
            return None

        return RuleDetection(
            issue=mk_issue(
                layer=self.layer,
                kind=self.kind,
                severity="info",
                message="Minor structure/semantic normalizations available (objective sense alignment, constraint sense tokens, binary bounds).",
                nodes=nodes,
            ),
            data=data,
        )

    def apply(self, ir: Any, detection: RuleDetection) -> Optional[Dict[str, Any]]:
        changed_fields: List[str] = []
        data = detection.data or {}

        # objective sense alignment
        meta = getattr(ir, "meta", None)
        obj = getattr(ir, "objective", None)
        ms = _lc(getattr(meta, "sense", "")) if meta is not None else ""
        os = _lc(getattr(obj, "sense", "")) if obj is not None else ""
        if ms in ("min", "max") and os in ("min", "max") and ms != os and obj is not None:
            setattr(obj, "sense", ms)
            changed_fields.append("objective.sense")

        # constraint sense normalization
        bad = set(data.get("bad_constraint_sense", []) or [])
        if bad:
            for c in _iter_constraints(ir):
                if getattr(c, "name", "") in bad:
                    s0 = getattr(c, "sense", "")
                    s1 = _SENSE_MAP.get(_lc(s0), None)
                    if s1 and s1 != s0:
                        setattr(c, "sense", s1)
                        changed_fields.append(f"constraints[{getattr(c,'name','?')}].sense")

        # binary bounds normalization
        badb = set(data.get("bad_binary_bounds", []) or [])
        if badb:
            for v in _iter_vars(ir):
                if getattr(v, "name", "") in badb:
                    if getattr(v, "lb", None) not in (0, 0.0):
                        setattr(v, "lb", 0.0)
                        changed_fields.append(f"vars[{getattr(v,'name','?')}].lb")
                    if getattr(v, "ub", None) not in (1, 1.0):
                        setattr(v, "ub", 1.0)
                        changed_fields.append(f"vars[{getattr(v,'name','?')}].ub")

        if not changed_fields:
            return None

        return mk_repair(
            layer=self.layer,
            kind=self.kind,
            message="Applied minor structure/semantic normalizations (objective sense, constraint sense tokens, binary bounds).",
            changed_fields=changed_fields,
        )


# -----------------------------------------------------------------------------
# Public factory
# -----------------------------------------------------------------------------

def get_layer2_rules() -> List[VerifierRule]:
    # Order: semantic-impactful rules first.
    return [
        IntegralitySanity(),
        ConstraintDirectionSanity(),
        GenericStructureSemanticSanity(),
    ]
