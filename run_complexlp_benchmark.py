# run_complexlp_benchmark.py
# Mamo_complex_lp_clean.jsonl -> ir2solve_pipeline -> CSV + IR JSON + trace.jsonl
# Dataset format (per line json):
#   {"Question": "...", "Answer": "...", ...}

from __future__ import annotations

import os
import json
import csv
import traceback
from typing import Any, Dict, Optional, Tuple, List

from openai import OpenAI
from ir2solve_pipeline import run_ir2solve_pipeline, PipelineConfig

# -------------------------
# Config
# -------------------------
DATASET_PATH = "data/Mamo/Mamo_complex_lp_clean.jsonl"

RESULT_DIR = "result_Mamo_complex"
IR_OUTPUT_DIR = os.path.join(RESULT_DIR, "ir_outputs_Mamo_complex")
RESULT_CSV_PATH = os.path.join(RESULT_DIR, "Mamo_complex_results.csv")
TRACE_JSONL_PATH = os.path.join(RESULT_DIR, "Mamo_complex_trace.jsonl")
SUMMARY_TXT_PATH = os.path.join(RESULT_DIR, "Mamo_complex_summary.txt")

LLM_MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.0
TIME_LIMIT_SEC = 60.0

# switches for ablation
LAYER1_ON = True
LAYER2_ON = True
LAYER3_ON = True
REPAIRS_ON = True
DETERMINE_ON = True

# -------------------------
# Utils
# -------------------------
def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def is_close(a: float, b: float, atol: float = 1e-4, rtol: float = 1e-6) -> bool:
    return abs(a - b) <= (atol + rtol * max(1.0, abs(b)))


def _safe_problem_id(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


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
# Per-instance solve
# -------------------------
def solve_one_instance(
    idx: int,
    question_text: str,
    gt_answer_raw: Any,
    client: OpenAI,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    instance_id = f"mamo_complex_lp_{idx}"
    gt_value = safe_float(gt_answer_raw)

    row: Dict[str, Any] = {
        "index": idx,
        "problem_id": instance_id,
        "status": "NONE",
        "obj_value": "",
        "ground_truth": "" if gt_value is None else float(gt_value),
        "correct": 0,
        "issues": "",
        "repairs": "",
        "failure_stage": "",
        "error": "",
    }

    ir_dict: Dict[str, Any] = {}
    trace: Dict[str, Any] = {}

    try:
        cfg = PipelineConfig(
            model_name=LLM_MODEL_NAME,
            temperature=float(TEMPERATURE),
            timelimit_sec=float(TIME_LIMIT_SEC),
            layer1_on=bool(LAYER1_ON),
            layer2_on=bool(LAYER2_ON),
            layer3_on=bool(LAYER3_ON),
            repairs_on=bool(REPAIRS_ON),
            determine_on=bool(DETERMINE_ON),
        )

        res = run_ir2solve_pipeline(
            question_text=question_text,
            client=client,
            config=cfg,
            problem_id=instance_id,
            meta_override={"source": "Mamo_complex_lp"},
        )

        ir_dict = getattr(res, "ir_dict", None) or {}
        trace = getattr(res, "trace", None) or {}

        row["failure_stage"] = getattr(res, "failure_stage", "") or ""
        row["error"] = getattr(res, "error", "") or ""

        status = getattr(res, "gurobi_status_name", None) or "NONE"
        row["status"] = status

        obj = getattr(res, "gurobi_obj_value", None)
        if obj is not None:
            row["obj_value"] = float(obj)

        vrep = getattr(res, "verifier_report", None) or {}
        row["issues"] = ";".join(_extract_kinds(vrep, "issues"))
        row["repairs"] = ";".join(_extract_kinds(vrep, "repairs"))

        if obj is not None and gt_value is not None:
            row["correct"] = int(is_close(float(obj), float(gt_value)))

        if isinstance(trace, dict):
            trace.setdefault("eval", {})
            trace["eval"].update(
                {
                    "ground_truth_raw": gt_answer_raw,
                    "ground_truth_value": gt_value,
                    "obj_value": obj,
                    "status": status,
                    "correct": row["correct"],
                }
            )

    except Exception as e:
        row["failure_stage"] = row["failure_stage"] or "benchmark_exception"
        row["error"] = f"{type(e).__name__}: {e}"
        trace = {
            "meta": {"problem_id": instance_id, "source": "Mamo_complex_lp"},
            "failure_stage": row["failure_stage"],
            "error": row["error"],
            "traceback": traceback.format_exc(limit=30),
        }

    return row, ir_dict, trace


# -------------------------
# Main
# -------------------------
def main() -> None:
    ensure_dir(RESULT_DIR)
    ensure_dir(IR_OUTPUT_DIR)

    client = OpenAI()

    fieldnames = [
        "index",
        "problem_id",
        "status",
        "obj_value",
        "ground_truth",
        "correct",
        "issues",
        "repairs",
        "failure_stage",
        "error",
    ]

    total = solved = correct_cnt = 0

    with open(RESULT_CSV_PATH, "w", newline="", encoding="utf-8") as f_csv, open(
        TRACE_JSONL_PATH, "w", encoding="utf-8"
    ) as f_trace:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        with open(DATASET_PATH, "r", encoding="utf-8") as f_in:
            for idx, line in enumerate(f_in):
                line = line.strip()
                if not line:
                    continue
                total += 1

                try:
                    record = json.loads(line)
                    question_text = record.get("Question", "") or ""
                    gt_answer_raw = record.get("Answer", "")
                except Exception as e:
                    row = {
                        "index": idx,
                        "problem_id": f"mamo_complex_lp_{idx}",
                        "status": "ERROR",
                        "obj_value": "",
                        "ground_truth": "",
                        "correct": 0,
                        "issues": "",
                        "repairs": "",
                        "failure_stage": "input_json_decode",
                        "error": f"{type(e).__name__}: {e}",
                    }
                    writer.writerow(row)
                    f_trace.write(
                        json.dumps(
                            {"meta": {"problem_id": row["problem_id"], "source": "Mamo_complex_lp"}, "error": row["error"]},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                row, ir_dict, trace = solve_one_instance(idx, question_text, gt_answer_raw, client)
                writer.writerow(row)
                f_trace.write(json.dumps(trace, ensure_ascii=False) + "\n")

                safe_id = _safe_problem_id(row["problem_id"])
                with open(os.path.join(IR_OUTPUT_DIR, f"{idx:03d}_{safe_id}.json"), "w", encoding="utf-8") as f_json:
                    json.dump(ir_dict, f_json, indent=2, ensure_ascii=False)

                if row["obj_value"] != "":
                    solved += 1
                if row["correct"]:
                    correct_cnt += 1

                print(
                    f"[{idx}] status={row['status']} obj={row['obj_value']} gt={row['ground_truth']} "
                    f"correct={row['correct']} issues={row['issues']} repairs={row['repairs']} "
                    f"stage={row['failure_stage']} err={row['error']}"
                )

    summary = [
        "====== SUMMARY ======",
        f"Total instances: {total}",
        f"Solved (got ObjVal): {solved}",
        f"Correct (ObjVal ≈ GT): {correct_cnt}",
    ]
    if total > 0:
        summary.append(f"Accuracy: {correct_cnt}/{total} = {correct_cnt/total:.3f}")
        summary.append(f"Solved ratio: {solved}/{total} = {solved/total:.3f}")

    print("\n" + "\n".join(summary))
    with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
