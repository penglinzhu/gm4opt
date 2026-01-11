# verifier_layer1.py
# Layer 1: Compile-safety & canonicalization
#
# Scope:
# 1) KeyError fixes: key canonicalization; 2D param canonicalization; missing diagonal fill.
# 2) NameError fixes: unroll constraints with free index symbols over inferred sets.
# 3) TypeError fixes: quicksum/sum call normalization.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import ast
import re

from ir2solve_ir import ConstraintDef
from ir2solve_verifier_core import (
    VerifierRule,
    RuleDetection,
    mk_issue,
    mk_repair,
    find_set,
    find_param,
    find_var,
    is_dict_of_dict,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_ALLOWED_GLOBALS = {
    "quicksum",
    "sum",
    "range",
    "len",
    "min",
    "max",
    "abs",
    "enumerate",
    "True",
    "False",
    "None",
}

# Common short index symbols that often appear as free indices in constraints.
_FREE_INDEX_CANDIDATES = {"i", "j", "k", "t", "u", "v", "n", "m", "p", "q", "w"}


def _iter_expr_fields(ir: Any) -> List[Tuple[str, Any, str]]:
    out: List[Tuple[str, Any, str]] = []
    obj = getattr(ir, "objective", None)
    if obj is not None and isinstance(getattr(obj, "expr", None), str):
        out.append(("objective.expr", obj, "expr"))

    for idx, c in enumerate(getattr(ir, "constraints", []) or []):
        if isinstance(getattr(c, "expr_lhs", None), str):
            out.append((f"constraints[{idx}].expr_lhs", c, "expr_lhs"))
        if isinstance(getattr(c, "expr_rhs", None), str):
            out.append((f"constraints[{idx}].expr_rhs", c, "expr_rhs"))
    return out


def _collect_defined_names(ir: Any) -> Dict[str, List[str]]:
    sets = [getattr(s, "name", "") for s in getattr(ir, "sets", []) or []]
    params = [getattr(p, "name", "") for p in getattr(ir, "params", []) or []]
    vars_ = [getattr(v, "name", "") for v in getattr(ir, "vars", []) or []]
    return {"sets": [x for x in sets if x], "params": [x for x in params if x], "vars": [x for x in vars_ if x]}


def _rewrite_all_expr(ir: Any, fn) -> List[str]:
    changed_fields: List[str] = []
    for path, obj, attr in _iter_expr_fields(ir):
        s = getattr(obj, attr, "")
        if not isinstance(s, str) or not s.strip():
            continue
        new_s, changed = fn(s)
        if changed and isinstance(new_s, str) and new_s != s:
            setattr(obj, attr, new_s)
            changed_fields.append(path)
    return changed_fields


def _collect_comprehension_targets(tree: ast.AST) -> set[str]:
    bound: set[str] = set()

    def _add_target(t: ast.AST) -> None:
        if isinstance(t, ast.Name):
            bound.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for elt in t.elts:
                _add_target(elt)

    class _V(ast.NodeVisitor):
        def visit_comprehension(self, node: ast.comprehension) -> None:
            _add_target(node.target)
            self.generic_visit(node)

    _V().visit(tree)
    return bound


def _extract_load_names(expr: str) -> Tuple[List[str], Optional[str], set[str]]:
    """Return (load_names, parse_error_msg, bound_names)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return [], f"SyntaxError: {e.msg} (line {e.lineno}, col {e.offset})", set()

    bound = _collect_comprehension_targets(tree)

    names: List[str] = []

    class _V(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                names.append(node.id)
            self.generic_visit(node)

    _V().visit(tree)
    return sorted(set(names)), None, bound


# -----------------------------------------------------------------------------
# Rule 1: Canonicalize set elements and param keys to strings (KeyError)
# -----------------------------------------------------------------------------

class CanonicalizeSetElementsAndParamKeys(VerifierRule):
    layer = "L1"
    kind = "canonicalize_set_and_keys"

    def detect(self, ir: Any) -> Optional[RuleDetection]:
        bad_sets: List[str] = []
        for s in getattr(ir, "sets", []) or []:
            elems = getattr(s, "elements", None)
            if isinstance(elems, list) and any(not isinstance(x, str) for x in elems):
                bad_sets.append(getattr(s, "name", ""))

        bad_param_keys = False
        for p in getattr(ir, "params", []) or []:
            idx = getattr(p, "indices", []) or []
            vals = getattr(p, "values", None)
            if not isinstance(vals, dict) or not idx:
                continue
            if len(idx) == 1:
                if any(not isinstance(k, str) for k in vals.keys()):
                    bad_param_keys = True
                    break
            if len(idx) == 2 and is_dict_of_dict(vals):
                if any(not isinstance(k, str) for k in vals.keys()):
                    bad_param_keys = True
                    break
                for row in vals.values():
                    if isinstance(row, dict) and any(not isinstance(k, str) for k in row.keys()):
                        bad_param_keys = True
                        break

        if not bad_sets and not bad_param_keys:
            return None

        return RuleDetection(
            issue=mk_issue(
                layer=self.layer,
                kind=self.kind,
                severity="warning",
                message="Non-string set elements or non-string param keys detected; canonicalize to reduce KeyError risk.",
                nodes=[x for x in bad_sets if x],
            ),
            data={"bad_sets": bad_sets, "bad_param_keys": bad_param_keys},
        )

    def apply(self, ir: Any, detection: RuleDetection) -> Optional[Dict[str, Any]]:
        changed_fields: List[str] = []

        # stringify set elements
        for s in getattr(ir, "sets", []) or []:
            elems = getattr(s, "elements", None)
            if not isinstance(elems, list):
                continue
            new_elems: List[str] = []
            changed = False
            for x in elems:
                if isinstance(x, str):
                    new_elems.append(x)
                else:
                    new_elems.append(str(x))
                    changed = True
            if changed:
                setattr(s, "elements", new_elems)
                changed_fields.append(f"sets[{getattr(s,'name','?')}].elements")

        # helper: stringify dict keys
        def _stringify_dict_keys(d: Dict[Any, Any]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for k, v in d.items():
                ks = k if isinstance(k, str) else str(k)
                if ks in out:
                    continue
                out[ks] = v
            return out

        # stringify param keys
        for p in getattr(ir, "params", []) or []:
            idx = getattr(p, "indices", []) or []
            vals = getattr(p, "values", None)
            if not isinstance(vals, dict) or not idx:
                continue

            if len(idx) == 1:
                new_vals = _stringify_dict_keys(vals)
                if new_vals != vals:
                    setattr(p, "values", new_vals)
                    changed_fields.append(f"params[{getattr(p,'name','?')}].values")

            elif len(idx) == 2 and is_dict_of_dict(vals):
                outer = _stringify_dict_keys(vals)
                inner_changed = False
                for ok, row in list(outer.items()):
                    if isinstance(row, dict):
                        new_row = _stringify_dict_keys(row)
                        if new_row != row:
                            outer[ok] = new_row
                            inner_changed = True
                if outer != vals or inner_changed:
                    setattr(p, "values", outer)
                    changed_fields.append(f"params[{getattr(p,'name','?')}].values")

        # rewrite numeric literal indices: x[1] -> x['1'] when digit-only sets exist
        digit_sets = set()
        for s in getattr(ir, "sets", []) or []:
            elems = getattr(s, "elements", []) or []
            if isinstance(elems, list) and elems and all(isinstance(e, str) and e.isdigit() for e in elems):
                digit_sets.add(getattr(s, "name", ""))

        def _rewrite_numeric_brackets(expr: str) -> Tuple[str, bool]:
            if not digit_sets:
                return expr, False
            pat = re.compile(r"\[(\d+)\]")
            new_expr = pat.sub(r"['\1']", expr)
            return new_expr, new_expr != expr

        changed_fields.extend(_rewrite_all_expr(ir, _rewrite_numeric_brackets))

        if not changed_fields:
            return None
        return mk_repair(
            layer=self.layer,
            kind=self.kind,
            message="Canonicalized set elements/param keys to strings; normalized numeric bracket indices.",
            changed_fields=changed_fields,
        )


# -----------------------------------------------------------------------------
# Rule 2: Canonicalize 2D param tuple-key dict -> nested dict (KeyError)
# -----------------------------------------------------------------------------

_TUPLE_KEY_PATTERNS = [
    re.compile(r"^\(\s*([^,]+)\s*,\s*([^)]+)\s*\)$"),  # "(a,b)"
    re.compile(r"^([^,]+)\s*,\s*([^,]+)$"),            # "a,b"
    re.compile(r"^([^|]+)\|\s*([^|]+)$"),              # "a|b"
]


def _strip_quotes(x: str) -> str:
    x = x.strip()
    if (x.startswith("'") and x.endswith("'")) or (x.startswith('"') and x.endswith('"')):
        return x[1:-1]
    return x


def _parse_tuple_key(k: Any) -> Optional[Tuple[str, str]]:
    if isinstance(k, tuple) and len(k) == 2:
        return (str(k[0]), str(k[1]))
    if not isinstance(k, str):
        return None
    s = k.strip()
    for pat in _TUPLE_KEY_PATTERNS:
        m = pat.match(s)
        if not m:
            continue
        a = _strip_quotes(m.group(1).strip())
        b = _strip_quotes(m.group(2).strip())
        return (str(a), str(b))
    return None


class Canonicalize2DParamToNestedDict(VerifierRule):
    layer = "L1"
    kind = "canonicalize_2d_param_values"

    def detect(self, ir: Any) -> Optional[RuleDetection]:
        offenders: List[str] = []
        for p in getattr(ir, "params", []) or []:
            idx = getattr(p, "indices", []) or []
            if len(idx) != 2:
                continue
            vals = getattr(p, "values", None)
            if not isinstance(vals, dict):
                continue
            if is_dict_of_dict(vals):
                continue
            if any(_parse_tuple_key(k) is not None for k in vals.keys()):
                offenders.append(getattr(p, "name", ""))
        if not offenders:
            return None
        return RuleDetection(
            issue=mk_issue(
                layer=self.layer,
                kind=self.kind,
                severity="warning",
                message="2D param appears to use tuple-key dict; canonicalize to nested dict.",
                nodes=[x for x in offenders if x],
            ),
            data={"offenders": offenders},
        )

    def apply(self, ir: Any, detection: RuleDetection) -> Optional[Dict[str, Any]]:
        changed_fields: List[str] = []
        offenders = set(detection.data.get("offenders", []) or [])
        for p in getattr(ir, "params", []) or []:
            if getattr(p, "name", "") not in offenders:
                continue
            vals = getattr(p, "values", None)
            if not isinstance(vals, dict) or is_dict_of_dict(vals):
                continue

            nested: Dict[str, Dict[str, Any]] = {}
            parsed_any = False
            for k, v in vals.items():
                ij = _parse_tuple_key(k)
                if ij is None:
                    continue
                i, j = ij
                nested.setdefault(str(i), {})[str(j)] = v
                parsed_any = True

            if parsed_any and nested:
                setattr(p, "values", nested)
                changed_fields.append(f"params[{getattr(p,'name','?')}].values")

        if not changed_fields:
            return None
        return mk_repair(
            layer=self.layer,
            kind=self.kind,
            message="Converted tuple-key 2D param dict to nested dict canonical form.",
            changed_fields=changed_fields,
        )


# -----------------------------------------------------------------------------
# Rule 3: Fill missing diagonal for square 2D params (KeyError)
# -----------------------------------------------------------------------------

class FillMissingDiagonalForSquare2DParams(VerifierRule):
    layer = "L1"
    kind = "fill_missing_diagonal"

    def detect(self, ir: Any) -> Optional[RuleDetection]:
        offenders: List[str] = []
        for p in getattr(ir, "params", []) or []:
            idx = getattr(p, "indices", []) or []
            if len(idx) != 2 or idx[0] != idx[1]:
                continue
            vals = getattr(p, "values", None)
            if not is_dict_of_dict(vals):
                continue
            s = find_set(ir, idx[0])
            elems = getattr(s, "elements", []) if s is not None else []
            if not isinstance(elems, list) or not elems:
                continue
            need = False
            for i in elems:
                if i not in vals:
                    need = True
                    break
                row = vals.get(i, {})
                if isinstance(row, dict) and i not in row:
                    need = True
                    break
            if need:
                offenders.append(getattr(p, "name", ""))
        if not offenders:
            return None
        return RuleDetection(
            issue=mk_issue(
                layer=self.layer,
                kind=self.kind,
                severity="warning",
                message="Square 2D param missing diagonal/rows; fill with 0.0 to avoid KeyError.",
                nodes=[x for x in offenders if x],
            ),
            data={"offenders": offenders},
        )

    def apply(self, ir: Any, detection: RuleDetection) -> Optional[Dict[str, Any]]:
        changed_fields: List[str] = []
        offenders = set(detection.data.get("offenders", []) or [])

        for p in getattr(ir, "params", []) or []:
            if getattr(p, "name", "") not in offenders:
                continue
            idx = getattr(p, "indices", []) or []
            if len(idx) != 2 or idx[0] != idx[1]:
                continue
            vals = getattr(p, "values", None)
            if not is_dict_of_dict(vals):
                continue
            s = find_set(ir, idx[0])
            elems = getattr(s, "elements", []) if s is not None else []
            if not isinstance(elems, list) or not elems:
                continue

            changed = False
            for i in elems:
                if i not in vals or not isinstance(vals.get(i), dict):
                    vals[i] = {}
                    changed = True
                row = vals[i]
                if i not in row:
                    row[i] = 0.0
                    changed = True

            if changed:
                setattr(p, "values", vals)
                changed_fields.append(f"params[{getattr(p,'name','?')}].values")

        if not changed_fields:
            return None
        return mk_repair(
            layer=self.layer,
            kind=self.kind,
            message="Filled missing diagonal entries (and missing rows) with 0.0 for square 2D params.",
            changed_fields=changed_fields,
        )


# -----------------------------------------------------------------------------
# Rule 4: Unroll free-index constraints over inferred sets (NameError)
# -----------------------------------------------------------------------------

def _sanitize_for_name(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x or "elem"


def _infer_set_for_index_symbol(ir: Any, sym: str, expr: str) -> Optional[str]:
    """
    Infer which set `sym` ranges over.

    Priority:
    1) Generator binding: "for sym in SetName"
    2) First-dim indexing: X[sym] -> indices[0]
    3) Second-dim indexing: X[...][sym] -> indices[1]
    """
    if not isinstance(expr, str) or not expr:
        return None

    m = re.search(rf"\bfor\s+{re.escape(sym)}\s+in\s+([A-Za-z_]\w*)\b", expr)
    if m:
        cand_set = m.group(1)
        if find_set(ir, cand_set) is not None:
            return cand_set

    counts: Dict[str, int] = {}

    pat1 = re.compile(rf"\b([A-Za-z_]\w*)\s*\[\s*{re.escape(sym)}\s*\]")
    for mm in pat1.finditer(expr):
        name = mm.group(1)
        v = find_var(ir, name)
        p = find_param(ir, name)
        idx = getattr(v, "indices", None) if v is not None else (getattr(p, "indices", None) if p is not None else None)
        if isinstance(idx, list) and len(idx) >= 1 and isinstance(idx[0], str) and idx[0]:
            counts[idx[0]] = counts.get(idx[0], 0) + 1

    pat2 = re.compile(rf"\b([A-Za-z_]\w*)\s*\[\s*[^]]+?\s*\]\s*\[\s*{re.escape(sym)}\s*\]")
    for mm in pat2.finditer(expr):
        name = mm.group(1)
        v = find_var(ir, name)
        p = find_param(ir, name)
        idx = getattr(v, "indices", None) if v is not None else (getattr(p, "indices", None) if p is not None else None)
        if isinstance(idx, list) and len(idx) >= 2 and isinstance(idx[1], str) and idx[1]:
            counts[idx[1]] = counts.get(idx[1], 0) + 1

    if not counts:
        return None

    best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(best) == 1:
        return best[0][0]
    if best[0][1] >= best[1][1] + 2:
        return best[0][0]
    return None


def _replace_index_symbol_with_literal(expr: str, sym: str, literal: str) -> str:
    # [sym] -> ['literal']
    return re.sub(rf"\[\s*{re.escape(sym)}\s*\]", f"['{literal}']", expr)


class UnrollFreeIndexConstraintsOverSet(VerifierRule):
    layer = "L1"
    kind = "unroll_free_index_constraints"
    MAX_UNROLL = 50

    def detect(self, ir: Any) -> Optional[RuleDetection]:
        defined = _collect_defined_names(ir)
        env_names = set(defined["sets"] + defined["params"] + defined["vars"]) | _ALLOWED_GLOBALS

        candidates: List[Dict[str, Any]] = []
        for idx, c in enumerate(getattr(ir, "constraints", []) or []):
            lhs = getattr(c, "expr_lhs", "")
            rhs = getattr(c, "expr_rhs", "")
            if not isinstance(lhs, str) or not isinstance(rhs, str):
                continue

            undefined: List[str] = []
            for s in (lhs, rhs):
                names, perr, bound = _extract_load_names(s)
                if perr is not None:
                    continue
                for n in names:
                    if n in bound:
                        continue
                    if n not in env_names:
                        undefined.append(n)
            undefined = sorted(set(undefined))
            if not undefined:
                continue

            hit_syms: List[str] = []
            for sym in undefined:
                if sym not in _FREE_INDEX_CANDIDATES:
                    continue
                if re.search(rf"\b{re.escape(sym)}\b", lhs) or re.search(rf"\b{re.escape(sym)}\b", rhs):
                    hit_syms.append(sym)
            if not hit_syms:
                continue

            sym_to_set: Dict[str, str] = {}
            for sym in hit_syms:
                set_name = _infer_set_for_index_symbol(ir, sym, lhs) or _infer_set_for_index_symbol(ir, sym, rhs)
                if not set_name:
                    continue
                s_obj = find_set(ir, set_name)
                elems = getattr(s_obj, "elements", []) if s_obj is not None else []
                if not isinstance(elems, list) or not elems:
                    continue
                if len(elems) > self.MAX_UNROLL:
                    continue
                sym_to_set[sym] = set_name

            if sym_to_set:
                candidates.append(
                    {
                        "constraint_index": idx,
                        "constraint_name": getattr(c, "name", f"c{idx}"),
                        "sym_to_set": sym_to_set,
                    }
                )

        if not candidates:
            return None

        names = [x["constraint_name"] for x in candidates]
        return RuleDetection(
            issue=mk_issue(
                layer=self.layer,
                kind=self.kind,
                severity="warning",
                message="Detected constraints with free index symbols; unroll over inferred sets to avoid NameError.",
                nodes=names,
            ),
            data={"candidates": candidates},
        )

    def apply(self, ir: Any, detection: RuleDetection) -> Optional[Dict[str, Any]]:
        candidates = detection.data.get("candidates", []) or []
        if not candidates:
            return None

        cand_by_idx = {c["constraint_index"]: c for c in candidates}
        new_constraints: List[Any] = []
        changed_fields: List[str] = []
        unrolled_names: List[str] = []

        for idx, c in enumerate(getattr(ir, "constraints", []) or []):
            if idx not in cand_by_idx:
                new_constraints.append(c)
                continue

            info = cand_by_idx[idx]
            sym_to_set: Dict[str, str] = info["sym_to_set"]
            cname = getattr(c, "name", f"c{idx}")

            lhs0 = getattr(c, "expr_lhs", "")
            rhs0 = getattr(c, "expr_rhs", "")
            sense0 = getattr(c, "sense", "==")
            desc0 = getattr(c, "description", None)

            # Choose ONE symbol to unroll: prefer a symbol not bound by a generator in this constraint.
            def _is_bound(sym: str, s: Any) -> bool:
                return isinstance(s, str) and (f"for {sym} in" in s)

            sym = None
            set_name = None
            for cand_sym, cand_set in sym_to_set.items():
                if (not _is_bound(cand_sym, lhs0)) and (not _is_bound(cand_sym, rhs0)):
                    sym, set_name = cand_sym, cand_set
                    break
            if sym is None:
                sym, set_name = list(sym_to_set.items())[0]

            s_obj = find_set(ir, set_name)
            elems = getattr(s_obj, "elements", []) if s_obj is not None else []
            if not isinstance(elems, list) or not elems or len(elems) > self.MAX_UNROLL:
                new_constraints.append(c)
                continue

            for e in elems:
                e_str = str(e)
                lhs = _replace_index_symbol_with_literal(lhs0, sym, e_str)
                rhs = _replace_index_symbol_with_literal(rhs0, sym, e_str)
                suffix = _sanitize_for_name(e_str)
                new_name = f"{cname}__{sym}_{suffix}"
                new_constraints.append(
                    ConstraintDef(
                        name=new_name,
                        expr_lhs=lhs,
                        sense=sense0,
                        expr_rhs=rhs,
                        description=desc0,
                    )
                )

            unrolled_names.append(cname)
            changed_fields.append(f"constraints[{idx}] ({cname}) -> unrolled over {set_name} by {sym}")

        if not unrolled_names:
            return None

        setattr(ir, "constraints", new_constraints)
        changed_fields.append("constraints (list replaced)")

        return mk_repair(
            layer=self.layer,
            kind=self.kind,
            message=f"Unrolled {len(unrolled_names)} constraint(s) over inferred sets: {unrolled_names}",
            changed_fields=changed_fields,
        )


# -----------------------------------------------------------------------------
# Rule 5: Fix quicksum/sum call typos (TypeError)
# - quicksum(expr, for i in I) -> quicksum(expr for i in I)
# - sum(expr, for i in I)      -> sum(expr for i in I)
# - quicksum(a, b, c)          -> (a) + (b) + (c)
# -----------------------------------------------------------------------------

class FixSumQuicksumCallTypos(VerifierRule):
    layer = "L1"
    kind = "fix_sum_calls"

    def detect(self, ir: Any) -> Optional[RuleDetection]:
        offenders: List[str] = []
        for path, obj, attr in _iter_expr_fields(ir):
            s = getattr(obj, attr, "")
            if not isinstance(s, str) or not s.strip():
                continue

            # Pattern A: comma before 'for' in sum/quicksum
            if re.search(r"\bquicksum\s*\(\s*[^)]*,\s*for\b", s) or re.search(r"\bsum\s*\(\s*[^)]*,\s*for\b", s):
                offenders.append(path)
                continue

            # Pattern B: quicksum with >=2 positional args and no generator
            if "quicksum" in s and " for " not in s and "," in s:
                try:
                    tree = ast.parse(s, mode="eval")
                except SyntaxError:
                    continue
                node = tree.body
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "quicksum"
                    and len(node.args) >= 2
                    and len(node.keywords) == 0
                ):
                    offenders.append(path)

        if not offenders:
            return None
        return RuleDetection(
            issue=mk_issue(
                layer=self.layer,
                kind=self.kind,
                severity="warning",
                message="Detected sum/quicksum call typos that may cause TypeError; rewrite to safe forms.",
                nodes=offenders,
            ),
            data={"offenders": offenders},
        )

    def apply(self, ir: Any, detection: RuleDetection) -> Optional[Dict[str, Any]]:
        offenders = set(detection.data.get("offenders", []) or [])
        changed_fields: List[str] = []

        def _fix_expr(expr: str) -> Tuple[str, bool]:
            s = expr

            # Fix comma-before-for: quicksum(a, for i in I) -> quicksum(a for i in I)
            s2 = re.sub(r"\bquicksum\s*\(\s*([^)]*?)\s*,\s*for\b", r"quicksum(\1 for", s)
            s3 = re.sub(r"\bsum\s*\(\s*([^)]*?)\s*,\s*for\b", r"sum(\1 for", s2)

            if s3 != s:
                return s3, True

            # Fix quicksum(a,b,...) -> (a)+(b)+...
            if "quicksum" not in s3 or " for " in s3 or "," not in s3:
                return s3, False
            try:
                tree = ast.parse(s3, mode="eval")
            except SyntaxError:
                return s3, False
            node = tree.body
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "quicksum"
                and len(node.args) >= 2
                and len(node.keywords) == 0
            ):
                return s3, False

            parts: List[str] = []
            for a in node.args:
                seg = None
                try:
                    seg = ast.get_source_segment(s3, a)
                except Exception:
                    seg = None
                if not seg:
                    try:
                        seg = ast.unparse(a)
                    except Exception:
                        return s3, False
                parts.append(f"({seg.strip()})")
            new_s = " + ".join(parts)
            return new_s, new_s != s3

        for path, obj, attr in _iter_expr_fields(ir):
            if path not in offenders:
                continue
            s = getattr(obj, attr, "")
            if not isinstance(s, str):
                continue
            new_s, changed = _fix_expr(s)
            if changed:
                setattr(obj, attr, new_s)
                changed_fields.append(path)

        if not changed_fields:
            return None
        return mk_repair(
            layer=self.layer,
            kind=self.kind,
            message="Rewrote sum/quicksum call typos into safe generator form or '+' chain.",
            changed_fields=changed_fields,
        )


# -----------------------------------------------------------------------------
# Public factory
# -----------------------------------------------------------------------------

def get_layer1_rules() -> List[VerifierRule]:
    # Minimal set of deterministic L1 repairs.
    return [
        CanonicalizeSetElementsAndParamKeys(),
        Canonicalize2DParamToNestedDict(),
        FillMissingDiagonalForSquare2DParams(),
        UnrollFreeIndexConstraintsOverSet(),
        FixSumQuicksumCallTypos(),
    ]
