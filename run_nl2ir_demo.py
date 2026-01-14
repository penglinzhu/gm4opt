# run_nl2ir_demo.py
# single-instance demo
# NL -> ir2solve_pipeline -> print/save artifacts

from __future__ import annotations

import os
import json
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List

from openai import OpenAI
from ir2solve_pipeline import PipelineConfig, run_ir2solve_pipeline

SAVE_ARTIFACTS = True
ARTIFACT_DIR = "demo_artifacts"


# -------------------------
# utils
# -------------------------
def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _try_asdict(x: Any) -> Any:
    if is_dataclass(x):
        try:
            return asdict(x)
        except Exception:
            return str(x)
    return x


def _json_safe(x: Any) -> Any:
    """Make obj JSON-serializable (tuple/set -> list; tuple keys -> str)."""
    if is_dataclass(x):
        try:
            x = asdict(x)
        except Exception:
            return str(x)

    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if isinstance(k, tuple):
                k = "->".join(map(str, k))
            elif not isinstance(k, (str, int, float, bool)) and k is not None:
                k = str(k)
            out[k] = _json_safe(v)
        return out

    if isinstance(x, (list, tuple, set)):
        return [_json_safe(t) for t in x]

    return x


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)


def _pretty(obj: Any) -> str:
    try:
        return json.dumps(_json_safe(obj), ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def _extract_kinds(report: Any, key: str) -> List[str]:
    out: List[str] = []
    if not isinstance(report, dict):
        return out
    for it in (report.get(key, []) or []):
        if isinstance(it, dict):
            k = it.get("kind")
            if k and k not in out:
                out.append(str(k))
        elif isinstance(it, str) and it and it not in out:
            out.append(it)
    return out


# -------------------------
# main
# -------------------------
def main() -> None:
    QUESTION_TEXT = """
Now we need to determine 4 out of 5 workers to each complete one of the four tasks. Since each worker has different skill sets, the amount of time required for each worker to complete each task is also different. The time required for each worker to complete each task is shown in Table 5-2.\nTable 5-2\n\\begin{tabular}{|c|c|c|c|c|}\n\\hline Task Time Required & $A$ & $B$ & $C$ & $D$ \\\\\n\\hline Worker & & & & \\\\\n\\hline I & 9 & 4 & 3 & 7 \\\\\n\\hline II & 4 & 6 & 5 & 6 \\\\\n\\hline III & 5 & 4 & 7 & 5 \\\\\n\\hline IV & 7 & 5 & 2 & 3 \\\\\n\\hline V & 10 & 6 & 7 & 4 \\\\\n\\hline\n\\end{tabular}\n\nTry to find a work assignment plan that minimizes the total working hours.
""".strip()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    problem_id = f"demo_{run_id}"

    # switches for ablations
    cfg = PipelineConfig(
        model_name="gpt-4o",
        temperature=0.0,
        timelimit_sec=60.0,
        layer1_on=True,
        layer2_on=True,
        layer3_on=True,
        repairs_on=True,
        determine_on=True,
    )

    out_dir = os.path.join(ARTIFACT_DIR, f"demo_{run_id}")
    if SAVE_ARTIFACTS:
        ensure_dir(out_dir)
        write_text(os.path.join(out_dir, "question.txt"), QUESTION_TEXT)
        write_json(os.path.join(out_dir, "pipeline_config.json"), _try_asdict(cfg))

    print(f"\n==== IR2Solve DEMO ({problem_id}) ====")
    print("Config:", _pretty(_try_asdict(cfg)))
    print("\nQuestion(first 400 chars):")
    print(QUESTION_TEXT[:400] + ("..." if len(QUESTION_TEXT) > 400 else ""))

    client = OpenAI()

    try:
        res = run_ir2solve_pipeline(
            question_text=QUESTION_TEXT,
            client=client,
            config=cfg,
            problem_id=problem_id,
            meta_override={"source": "demo"},
        )
    except Exception as e:
        print("\n[PIPELINE EXCEPTION]")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc(limit=20)
        if SAVE_ARTIFACTS:
            write_json(
                os.path.join(out_dir, "summary.json"),
                {"problem_id": problem_id, "failure_stage": "pipeline_exception", "error": f"{type(e).__name__}: {e}"},
            )
        return

    failure_stage = getattr(res, "failure_stage", "") or ""
    error = getattr(res, "error", "") or ""
    print("\n[STATUS]")
    print("failure_stage:", failure_stage)
    print("error:", error)

    # --- IR dict ---
    ir_dict = getattr(res, "ir_dict", None) or {}
    if SAVE_ARTIFACTS:
        write_json(os.path.join(out_dir, "ir_dict.json"), ir_dict)

    # --- ModelIR counts ---
    ir = getattr(res, "ir", None)
    if ir is None:
        print("\n[MODELIR] None (failed at ir_parse or earlier).")
        sets_n = params_n = vars_n = cons_n = 0
    else:
        sets_n = len(getattr(ir, "sets", []) or [])
        params_n = len(getattr(ir, "params", []) or [])
        vars_n = len(getattr(ir, "vars", []) or [])
        cons_n = len(getattr(ir, "constraints", []) or [])
        print("\n[MODELIR COUNTS]")
        print(f"sets={sets_n}, params={params_n}, vars={vars_n}, constraints={cons_n}")

    # --- Verifier kinds ---
    verifier_report = getattr(res, "verifier_report", None) or {}
    issues_kinds = _extract_kinds(verifier_report, "issues")
    repairs_kinds = _extract_kinds(verifier_report, "repairs")
    print("\n[VERIFIER KINDS]")
    print("issues:", issues_kinds)
    print("repairs:", repairs_kinds)

    if SAVE_ARTIFACTS:
        write_json(
            os.path.join(out_dir, "verifier_kinds.json"),
            {"issues": issues_kinds, "repairs": repairs_kinds},
        )

    # --- Solver result ---
    status_name = getattr(res, "gurobi_status_name", None)
    obj_value = getattr(res, "gurobi_obj_value", None)
    print("\n[SOLVER]")
    print("status_name:", status_name)
    print("obj_value:", obj_value)

    # --- Trace ---
    trace = getattr(res, "trace", None) or {}
    if SAVE_ARTIFACTS:
        write_json(os.path.join(out_dir, "trace.json"), trace)

    # --- Summary ---
    summary = {
        "problem_id": problem_id,
        "failure_stage": failure_stage,
        "error": error,
        "modelir_counts": {"sets": sets_n, "params": params_n, "vars": vars_n, "constraints": cons_n},
        "verifier_issues": issues_kinds,
        "verifier_repairs": repairs_kinds,
        "solver_status_name": status_name,
        "obj_value": obj_value,
        "artifacts_dir": out_dir if SAVE_ARTIFACTS else None,
    }
    print("\n[SUMMARY]")
    print(_pretty(summary))

    if SAVE_ARTIFACTS:
        write_json(os.path.join(out_dir, "summary.json"), summary)
        print(f"\n[Artifacts saved] {out_dir}")


if __name__ == "__main__":
    main()
