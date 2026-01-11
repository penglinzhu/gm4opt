# verifier_core.py
# - Deterministic rule protocol: detect() -> issue + data; apply() -> repair (optional).

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import copy


# -----------------------------------------------------------------------------
# Config / Report helpers
# -----------------------------------------------------------------------------
@dataclass
class VerifierConfig:
    # switches for ablation
    layer1_on: bool = True
    layer2_on: bool = True
    layer3_on: bool = True
    repairs_on: bool = True  # if False: report-only, do not mutate original IR


def mk_issue(
    layer: str,
    kind: str,
    severity: str,
    message: str,
    nodes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "layer": layer,
        "kind": kind,
        "severity": severity,
        "message": message,
        "nodes": nodes or [],
    }


def mk_repair(
    layer: str,
    kind: str,
    message: str,
    changed_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "layer": layer,
        "kind": kind,
        "message": message,
        "changed_fields": changed_fields or [],
    }


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def find_set(ir: Any, name: str) -> Optional[Any]:
    for s in getattr(ir, "sets", []) or []:
        if getattr(s, "name", None) == name:
            return s
    return None


def find_param(ir: Any, name: str) -> Optional[Any]:
    for p in getattr(ir, "params", []) or []:
        if getattr(p, "name", None) == name:
            return p
    return None


def find_var(ir: Any, name: str) -> Optional[Any]:
    for v in getattr(ir, "vars", []) or []:
        if getattr(v, "name", None) == name:
            return v
    return None


def is_dict_of_dict(x: Any) -> bool:
    return isinstance(x, dict) and bool(x) and all(isinstance(v, dict) for v in x.values())


def quote_key(k: Any) -> str:
    """Render a set element key inside an expression string (safe for strings/numbers/tuples)."""
    return repr(k)


def fresh_var_name(ir: Any, base: str) -> str:
    """Generate a fresh variable name not colliding with existing vars."""
    existing = {getattr(v, "name", "") for v in getattr(ir, "vars", []) or []}
    if base not in existing:
        return base
    i = 1
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


# -----------------------------------------------------------------------------
# Rule protocol + runner
# -----------------------------------------------------------------------------
@dataclass
class RuleDetection:
    issue: Dict[str, Any]
    data: Dict[str, Any]


class VerifierRule:
    layer: str = "L?"
    kind: str = "unknown"

    def detect(self, ir: Any) -> Optional[RuleDetection]:
        raise NotImplementedError

    def apply(self, ir: Any, detection: RuleDetection) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


def run_rules(
    ir: Any,
    rules: List[VerifierRule],
    repairs_on: bool,
    issues: List[Dict[str, Any]],
    repairs: List[Dict[str, Any]],
) -> bool:
    """
    Run a list of rules in order.
    Returns whether IR changed (only meaningful when repairs_on=True).
    """
    changed = False
    for rule in rules:
        det = rule.detect(ir)
        if det is None:
            continue

        issues.append(det.issue)

        if repairs_on:
            rep = rule.apply(ir, det)
            if rep is not None:
                repairs.append(rep)
                changed = True
    return changed


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def run_verifier(ir: Any, config: Optional[VerifierConfig] = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Returns: (possibly modified IR, verifier_report)
    - If repairs_on=False: verifier runs on a deepcopy and returns original IR unchanged.
    """
    if config is None:
        config = VerifierConfig()

    working_ir = ir if config.repairs_on else copy.deepcopy(ir)

    issues: List[Dict[str, Any]] = []
    repairs: List[Dict[str, Any]] = []

    layer_ran = {"L1": False, "L2": False, "L3": False}
    layer_changed = {"L1": False, "L2": False, "L3": False}

    try:
        if config.layer1_on:
            from ir2solve_verifier_layer1 import get_layer1_rules
        if config.layer2_on:
            from ir2solve_verifier_layer2 import get_layer2_rules
        if config.layer3_on:
            from ir2solve_verifier_layer3 import get_layer3_rules

        if config.layer1_on:
            layer_ran["L1"] = True
            layer_changed["L1"] = run_rules(working_ir, get_layer1_rules(), config.repairs_on, issues, repairs)

        if config.layer2_on:
            layer_ran["L2"] = True
            layer_changed["L2"] = run_rules(working_ir, get_layer2_rules(), config.repairs_on, issues, repairs)

        if config.layer3_on:
            layer_ran["L3"] = True
            layer_changed["L3"] = run_rules(working_ir, get_layer3_rules(), config.repairs_on, issues, repairs)

        report_ok = True

    except Exception as e:
        report_ok = False
        issues.append(
            mk_issue(
                layer="VERIFIER",
                kind="verifier_exception",
                severity="error",
                message=f"{type(e).__name__}: {e}",
                nodes=[],
            )
        )

    report: Dict[str, Any] = {
        "ok": report_ok,
        "config": asdict(config),
        "layers": {
            "L1": {"ran": layer_ran["L1"], "changed_ir": layer_changed["L1"] if config.repairs_on else False},
            "L2": {"ran": layer_ran["L2"], "changed_ir": layer_changed["L2"] if config.repairs_on else False},
            "L3": {"ran": layer_ran["L3"], "changed_ir": layer_changed["L3"] if config.repairs_on else False},
        },
        "repairs_on": bool(config.repairs_on),
        "issues": issues,
        "repairs": repairs if config.repairs_on else [],
        "notes": "layered verifier",
    }

    return (working_ir if config.repairs_on else ir), report
