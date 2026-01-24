# run_nl4opt_benchmark.py
# Batch run NL4Opt directory dataset (each subdir = one instance)


from __future__ import annotations

import os
import json
import csv
import traceback
from typing import Any, Dict, Optional, List, Tuple

from openai import OpenAI
from ir2solve_pipeline import run_ir2solve_pipeline, PipelineConfig


# -------------------------
# LLM usage tracking
# -------------------------
class LLMUsageTracker:
    """Accumulate LLM call count and token usage from OpenAI SDK responses."""

    def __init__(self) -> None:
        self.calls: int = 0  # number of attempted API calls
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0

    def add_usage_from_response(self, resp: Any) -> None:
        """Extract usage fields from a response object and accumulate."""
        usage = getattr(resp, "usage", None)
        if usage is None:
            return

        # usage could be dict-like
        if isinstance(usage, dict):
            self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            self.total_tokens += int(usage.get("total_tokens", 0) or 0)
            return

        # or object-like
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        self.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)


def _wrap_openai_method(method, tracker: LLMUsageTracker):
    """Wrap an OpenAI SDK method to count calls and accumulate token usage."""
    def wrapper(*args, **kwargs):
        tracker.calls += 1
        resp = method(*args, **kwargs) 
        tracker.add_usage_from_response(resp)
        return resp
    return wrapper


def attach_llm_usage_tracker(client: OpenAI, tracker: LLMUsageTracker) -> None:
    """
    Monkey-patch common OpenAI SDK call sites so we can count:
      - total call attempts
      - prompt / completion / total tokens (when provided in response.usage)

    This is intentionally broad because different pipelines may call different endpoints.
    """
    # chat.completions.create
    try:
        if hasattr(client, "chat") and hasattr(client.chat, "completions") and hasattr(client.chat.completions, "create"):
            client.chat.completions.create = _wrap_openai_method(client.chat.completions.create, tracker)  # type: ignore
    except Exception:
        pass

    # beta.chat.completions.parse (structured outputs)
    try:
        if hasattr(client, "beta") and hasattr(client.beta, "chat") and hasattr(client.beta.chat, "completions") and hasattr(client.beta.chat.completions, "parse"):
            client.beta.chat.completions.parse = _wrap_openai_method(client.beta.chat.completions.parse, tracker)  # type: ignore
    except Exception:
        pass

    # responses.create (newer endpoint)
    try:
        if hasattr(client, "responses") and hasattr(client.responses, "create"):
            client.responses.create = _wrap_openai_method(client.responses.create, tracker)  # type: ignore
    except Exception:
        pass


# -------------------------
# Config
# -------------------------

# Dataset root dir: contains subfolders, each with description.txt and sample.json
NL4OPT_ROOT_DIR = "data/NL4Opt"

# Result root dir
RESULT_DIR = "result_NL4Opt"

# IR JSON output dir
IR_OUTPUT_DIR = os.path.join(RESULT_DIR, "ir_outputs_NL4Opt")

# CSV results
RESULT_CSV_PATH = os.path.join(RESULT_DIR, "NL4Opt_results.csv")

# Trace jsonl
TRACE_JSONL_PATH = os.path.join(RESULT_DIR, "NL4Opt_trace.jsonl")

# Summary txt
SUMMARY_TXT_PATH = os.path.join(RESULT_DIR, "NL4Opt_summary.txt")

# LLM model name
LLM_MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.0
TIME_LIMIT_SEC = 60.0

# switches for ablation
LAYER1_ON = True
LAYER2_ON = True
LAYER3_ON = True
REPAIRS_ON = True

# File conventions inside each problem dir
DESC_FILENAME = "description.txt"
GT_FILENAME = "sample.json"   
GT_FIELD = "output"           


# -------------------------
# Utils
# -------------------------
def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def safe_float(x: Any) -> Optional[float]:
    """Try convert x to float; return None on failure."""
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
# Dataset reading & validation
# -------------------------
def _list_problem_dirs(root_dir: str) -> List[Tuple[str, str]]:
    """
    List NL4Opt problem subdirs.
    Keep ONLY subdirs that contain both description.txt and sample.json.
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"NL4OPT_ROOT_DIR not found or not a directory: {root_dir}")

    items: List[Tuple[str, str]] = []
    for name in os.listdir(root_dir):
        abs_path = os.path.join(root_dir, name)
        if not os.path.isdir(abs_path):
            continue
        desc_path = os.path.join(abs_path, DESC_FILENAME)
        gt_path = os.path.join(abs_path, GT_FILENAME)
        if os.path.isfile(desc_path) and os.path.isfile(gt_path):
            items.append((name, abs_path))

    def sort_key(x: Tuple[str, str]):
        name = x[0]
        try:
            return (0, int(name))
        except ValueError:
            return (1, name)

    items.sort(key=sort_key)
    return items


def _read_description(desc_path: str) -> str:
    with open(desc_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _read_ground_truth_output(gt_path: str) -> Any:
    """
    NL4Opt sample.json example:
    [
        {
            "input": {...},
            "output": [5050]
        }
    ]
    Return the raw output object (often a list like [5050]) or a scalar.
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # sample.json is a list
    if isinstance(data, list):
        if not data:
            return None
        first = data[0]
        if not isinstance(first, dict):
            return None
        out = first.get(GT_FIELD, None)
    elif isinstance(data, dict):
        # just in case some samples are dict-shaped
        out = data.get(GT_FIELD, None)
    else:
        return None

    # output is typically a list like [5050]
    if isinstance(out, list):
        if not out:
            return None
        return out[0]
    return out


# -------------------------
# Per-instance solve
# -------------------------
def solve_one_instance(
    idx: int,
    problem_dir_name: str,
    question_text: str,
    gt_output_raw: Any,
    client: OpenAI,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    instance_id = f"NL4Opt_{problem_dir_name}"
    gt_value = safe_float(gt_output_raw)

    row: Dict[str, Any] = {
        "index": idx,
        "problem_id": instance_id,
        "problem_dir": problem_dir_name,
        "status": "NONE",
        "obj_value": "",
        "ground_truth": "" if gt_value is None else float(gt_value),
        "correct": 0,
        "issues": "",
        "repairs": "",
        "failure_stage": "",
        "error": "",
    }

    trace: Dict[str, Any] = {}
    ir_dict: Dict[str, Any] = {}

    try:
        config = PipelineConfig(
            model_name=LLM_MODEL_NAME,
            temperature=TEMPERATURE,
            timelimit_sec=TIME_LIMIT_SEC,
            layer1_on=LAYER1_ON,
            layer2_on=LAYER2_ON,
            layer3_on=LAYER3_ON,
            repairs_on=REPAIRS_ON,
        )

        res = run_ir2solve_pipeline(
            question_text=question_text,
            client=client,
            config=config,
            problem_id=instance_id,
            meta_override={"source": "NL4Opt", "problem_dir": problem_dir_name},
        )

        trace = getattr(res, "trace", None) or {}
        ir_dict = getattr(res, "ir_dict", None) or {}

        status = getattr(res, "gurobi_status_name", None)
        if status is not None:
            row["status"] = str(status)

        if getattr(res, "failure_stage", None):
            row["failure_stage"] = str(res.failure_stage)

        if getattr(res, "error", None):
            row["error"] = str(res.error)

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
                    "ground_truth_raw": gt_output_raw,
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
            "meta": {"problem_id": instance_id, "source": "NL4Opt", "problem_dir": problem_dir_name},
            "failure_stage": row["failure_stage"],
            "error": row["error"],
            "traceback": traceback.format_exc(limit=30),
        }

    return row, trace, ir_dict


# -------------------------
# Main loop
# -------------------------
def main() -> None:
    ensure_dir(RESULT_DIR)
    ensure_dir(IR_OUTPUT_DIR)

    client = OpenAI()

    # attach tracker (counts all LLM calls across pipeline)
    llm_tracker = LLMUsageTracker()
    attach_llm_usage_tracker(client, llm_tracker)

    problem_dirs = _list_problem_dirs(NL4OPT_ROOT_DIR)
    print(f"[INFO] Found {len(problem_dirs)} instances under {NL4OPT_ROOT_DIR}")

    fieldnames = [
        "index",
        "problem_id",
        "problem_dir",
        "status",
        "obj_value",
        "ground_truth",
        "correct",
        "issues",
        "repairs",
        "failure_stage",
        "error",
    ]

    total = 0
    solved = 0
    correct_cnt = 0

    with open(RESULT_CSV_PATH, "w", newline="", encoding="utf-8") as f_csv, open(
        TRACE_JSONL_PATH, "w", encoding="utf-8"
    ) as f_trace:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (dir_name, abs_dir) in enumerate(problem_dirs):
            total += 1
            print(f"\n==== Solving instance #{idx} (dir {dir_name}) ====")

            desc_path = os.path.join(abs_dir, DESC_FILENAME)
            gt_path = os.path.join(abs_dir, GT_FILENAME)

            try:
                question_text = _read_description(desc_path)
            except Exception as e:
                row = {
                    "index": idx,
                    "problem_id": f"NL4Opt_{dir_name}",
                    "problem_dir": dir_name,
                    "status": "ERROR",
                    "obj_value": "",
                    "ground_truth": "",
                    "correct": 0,
                    "issues": "",
                    "repairs": "",
                    "failure_stage": "read_description",
                    "error": f"ReadDescriptionError: {type(e).__name__}: {e}",
                }
                writer.writerow(row)
                f_trace.write(
                    json.dumps(
                        {
                            "meta": {"problem_id": row["problem_id"], "source": "NL4Opt", "problem_dir": dir_name},
                            "failure_stage": "read_description",
                            "error": row["error"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            try:
                gt_output_raw = _read_ground_truth_output(gt_path)
            except Exception as e:
                row = {
                    "index": idx,
                    "problem_id": f"NL4Opt_{dir_name}",
                    "problem_dir": dir_name,
                    "status": "ERROR",
                    "obj_value": "",
                    "ground_truth": "",
                    "correct": 0,
                    "issues": "",
                    "repairs": "",
                    "failure_stage": "read_ground_truth",
                    "error": f"ReadGTError: {type(e).__name__}: {e}",
                }
                writer.writerow(row)
                f_trace.write(
                    json.dumps(
                        {
                            "meta": {"problem_id": row["problem_id"], "source": "NL4Opt", "problem_dir": dir_name},
                            "failure_stage": "read_ground_truth",
                            "error": row["error"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            row, trace, ir_dict = solve_one_instance(
                idx=idx,
                problem_dir_name=dir_name,
                question_text=question_text,
                gt_output_raw=gt_output_raw,
                client=client,
            )

            writer.writerow(row)

            # Update counters
            if row.get("obj_value", "") != "":
                solved += 1
            if int(row.get("correct", 0)) == 1:
                correct_cnt += 1

            # Save IR json for debugging
            if isinstance(ir_dict, dict) and ir_dict:
                ir_path = os.path.join(IR_OUTPUT_DIR, f"{_safe_problem_id(row['problem_id'])}.json")
                try:
                    with open(ir_path, "w", encoding="utf-8") as f_ir:
                        json.dump(ir_dict, f_ir, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            # Write trace
            if not isinstance(trace, dict):
                trace = {"meta": {"problem_id": row["problem_id"], "source": "NL4Opt", "problem_dir": dir_name}}
            f_trace.write(json.dumps(trace, ensure_ascii=False) + "\n")

    # Summary
    summary = [
        "====== SUMMARY ======",
        f"Total instances: {total}",
        f"Solved (got ObjVal): {solved}",
        f"Correct (ObjVal ≈ GT): {correct_cnt}",
    ]
    if total > 0:
        summary.append(f"Accuracy: {correct_cnt}/{total} = {correct_cnt/total:.3f}")
        summary.append(f"Solved ratio: {solved}/{total} = {solved/total:.3f}")

    # --- LLM usage statistics (total and per-problem averages) ---
    if total > 0:
        avg_calls = llm_tracker.calls / total
        avg_prompt = llm_tracker.prompt_tokens / total
        avg_completion = llm_tracker.completion_tokens / total
        avg_total = llm_tracker.total_tokens / total

        summary.append("====== LLM USAGE ======")
        summary.append(f"LLM calls: {llm_tracker.calls}")
        summary.append(
            f"Tokens: total={llm_tracker.total_tokens} (prompt={llm_tracker.prompt_tokens}, completion={llm_tracker.completion_tokens})"
        )
        summary.append(
            f"Avg per problem: calls={avg_calls:.6f}, tokens={avg_total:.2f} (prompt={avg_prompt:.2f}, completion={avg_completion:.2f})"
        )

    print("\n" + "\n".join(summary))
    with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
