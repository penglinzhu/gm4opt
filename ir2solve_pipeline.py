# ir2solve_pipeline.py
# NL -> LLM -> JSON IR -> ModelIR -> verifier (L1/L2/L3) -> Gurobi

from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import Any, Dict, Optional, List, Tuple

from openai import OpenAI
from gurobipy import GRB

from ir2solve_ir import ModelIR, ir_to_gurobi
from ir2solve_nl2ir import (
    build_system_prompt,
    build_user_prompt,
    extract_json_from_text,
    json_to_model_ir,
)
from ir2solve_verifier_core import run_verifier, VerifierConfig

from ir2solve_estimator import build_check_message, load_model_from_dict_str

from llm2code import build_generate_message, llm_to_gurobi


# -----------------------------------------------------------------------------
# Config / Result
# -----------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    # runtime params
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    timelimit_sec: float = 60.0

    # switches for ablation
    layer1_on: bool = True
    layer2_on: bool = True
    layer3_on: bool = True
    repairs_on: bool = True
    determine_on: bool = True
    estimate_on: bool = True


@dataclass
class PipelineResult:
    ir_dict: Dict[str, Any]
    ir: Optional[ModelIR]

    failure_stage: str
    error: str

    gurobi_status: Optional[int]
    gurobi_status_name: str
    gurobi_obj_value: Optional[float]

    verifier_report: Dict[str, Any]
    trace: Dict[str, Any]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_meta_fill(data: Dict[str, Any], problem_id: Optional[str], meta_override: Optional[Dict[str, Any]]) -> None:
    """
    Ensures meta exists and sets problem_id when provided.
    """
    if "meta" not in data or not isinstance(data.get("meta"), dict):
        data["meta"] = {}
    if meta_override:
        for k, v in meta_override.items():
            data["meta"][k] = v
    if problem_id:
        data["meta"]["problem_id"] = problem_id
    if not data["meta"].get("problem_id"):
        data["meta"]["problem_id"] = "ir2solve_instance"


def _build_verifier_config(cfg: PipelineConfig) -> VerifierConfig:
    """
    Pass fields that exist on VerifierConfig.
    """
    kwargs = {
        "layer1_on": cfg.layer1_on,
        "layer2_on": cfg.layer2_on,
        "layer3_on": cfg.layer3_on,
        "repairs_on": cfg.repairs_on,
    }
    if is_dataclass(VerifierConfig):
        fset = set(VerifierConfig.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in kwargs.items() if k in fset}
    return VerifierConfig(**kwargs)


def _status_name(code: int) -> str:
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return status_map.get(code, str(code))


def _extract_kinds(report: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    issues_k: List[str] = []
    repairs_k: List[str] = []
    if isinstance(report, dict):
        for it in report.get("issues", []) or []:
            if isinstance(it, dict) and it.get("kind"):
                issues_k.append(it["kind"])
        for it in report.get("repairs", []) or []:
            if isinstance(it, dict) and it.get("kind"):
                repairs_k.append(it["kind"])
    return issues_k, repairs_k


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run_ir2solve_pipeline(
    question_text: str,
    client: Optional[OpenAI] = None,
    config: Optional[PipelineConfig] = None,
    problem_id: Optional[str] = None,
    meta_override: Optional[Dict[str, Any]] = None,
) -> PipelineResult:
    if config is None:
        config = PipelineConfig()
    if client is None:
        client = create_client(config, test_flag=True)

    failure_stage = ""
    error = ""

    data: Dict[str, Any] = {}
    ir: Optional[ModelIR] = None
    verifier_report: Dict[str, Any] = {"ok": False, "issues": [], "repairs": []}

    gurobi_status: Optional[int] = None
    gurobi_status_name: str = "NONE"
    gurobi_obj_value: Optional[float] = None

    # --- 1) prompts ---
    system_prompt = ""
    user_prompt = ""
    try:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(question_text)
    except Exception as e:
        failure_stage = "build_prompts"
        error = f"{type(e).__name__}: {e}"

    # --- 2) LLM ---
    raw_llm_text = ""
    if not failure_stage:
        try:
            completion = client.chat.completions.create(
                model=config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config.temperature,
            )
            raw_llm_text = completion.choices[0].message.content or ""
        except Exception as e:
            failure_stage = "llm_call"
            error = f"{type(e).__name__}: {e}"

    # --- 3) JSON extract ---
    if not failure_stage:
        try:
            data = extract_json_from_text(raw_llm_text)
        except Exception as e:
            failure_stage = "json_extract"
            error = f"{type(e).__name__}: {e}"
            data = {}

    # --- 4) meta fill (best-effort, never fails) ---
    try:
        _safe_meta_fill(data, problem_id, meta_override)
    except Exception:
        pass

    try:
        if "meta" not in data or not isinstance(data.get("meta"), dict):
            data["meta"] = {}
        full_text = (question_text or "").strip()
        if full_text:
            data["meta"]["description"] = full_text[:4000]
    except Exception:
        pass

    # --- 5) JSON -> ModelIR ---
    if not failure_stage:
        try:
            ir = json_to_model_ir(data)
        except Exception as e:
            failure_stage = "ir_parse"
            error = f"{type(e).__name__}: {e}"
            ir = None

    # --- 6) verifier ---
    if not failure_stage and ir is not None:
        try:
            vcfg = _build_verifier_config(config)
            ir, verifier_report = run_verifier(ir, config=vcfg)
        except Exception as e:
            failure_stage = "verifier"
            error = f"{type(e).__name__}: {e}"

    # --- 7) Gurobi build + optimize ---
    if not failure_stage and ir is not None:
        if config.determine_on:
            # 确定性构造Gurobi模型
            try:
                model = ir_to_gurobi(ir)
            except Exception as e:
                failure_stage = "solver_build"
                error = f"{type(e).__name__}: {e}"
                model = None
        else:
            try:
                generate_messages = build_generate_message(question_text, ir)
                # call LLM to build Gurobi model
                generate_completion = client.chat.completions.create(
                    model=config.model_name,
                    messages=generate_messages,
                    temperature=config.temperature,
                )
                model_generate_code: str = generate_completion.choices[0].message.content or ""
                model = llm_to_gurobi(model_generate_code)
            except Exception as e:
                failure_stage = "solver_build"
                error = f"{type(e).__name__}: {e}"
                model = None

        if not failure_stage and model is not None:
            try:
                model.setParam("TimeLimit", float(config.timelimit_sec))
                model.optimize()

                gurobi_status = int(model.Status)
                gurobi_status_name = _status_name(gurobi_status)

                if model.Status == GRB.OPTIMAL:
                    gurobi_obj_value = float(model.ObjVal)
            except Exception as e:
                failure_stage = "solver_optimize"
                error = f"{type(e).__name__}: {e}"

    pid = (data.get("meta") or {}).get("problem_id", problem_id or "ir2solve_instance")
    issues_kinds, repairs_kinds = _extract_kinds(verifier_report)

    # --- 8) self-check + rebuild ---
    if not failure_stage and ir is not None:
        if config.estimate_on:
            try:
                # call LLM for confidence check
                check_messages = build_check_message(question_text, ir, gurobi_model=model)
                check_completion = client.chat.completions.create(
                    model=config.model_name,
                    messages=check_messages,
                    temperature=config.temperature,
                )
                raw_check_text: str = check_completion.choices[0].message.content or ""
                check_data = extract_json_from_text(raw_check_text)
            except Exception as e:
                failure_stage = "estimator_llm"
                error = f"{type(e).__name__}: {e}"
                check_data = {}
            
            confidence_score = check_data.get("confidence_score", 0)
            corrected_model_dict_str = check_data.get("corrected_model")
            print("confidence_score:", confidence_score)

            if not failure_stage and corrected_model_dict_str:
                try:
                    # print("corrected_model:", corrected_model_dict_str)
                    # print("reasoning:", check_data.get("reasoning", ""))
                    try:
                        # rebuild corrected model from LLM output
                        corrected_model = load_model_from_dict_str(corrected_model_dict_str)
                    except Exception as e:
                        print("Failed to rebuild corrected model:", str(e))
                        corrected_model = None
                    
                    if corrected_model:
                        print("Re-optimizing corrected model...")
                        corrected_model.setParam("TimeLimit", float(config.timelimit_sec))
                        corrected_model.optimize()
                        if corrected_model.Status == GRB.OPTIMAL:
                            gurobi_obj_value = float(corrected_model.ObjVal)
                except Exception as e:
                    failure_stage = "estimator_rebuild"
                    error = f"{type(e).__name__}: {e}"

    trace: Dict[str, Any] = {
        "meta": {
            "problem_id": pid,
            "model_name": config.model_name,
            "timelimit_sec": config.timelimit_sec,
            "layer1_on": config.layer1_on,
            "layer2_on": config.layer2_on,
            "layer3_on": config.layer3_on,
            "repairs_on": config.repairs_on,
        },
        "failure_stage": failure_stage,
        "error": error,
        "verifier": {
            "ok": bool(verifier_report.get("ok", False)) if isinstance(verifier_report, dict) else False,
            "issues": issues_kinds,
            "repairs": repairs_kinds,
        },
        "solver": {
            "status_name": gurobi_status_name,
            "obj_value": gurobi_obj_value,
        },
        # keep IR dict for replay
        "ir_dict": data,
    }

    return PipelineResult(
        ir_dict=data,
        ir=ir,
        failure_stage=failure_stage,
        error=error,
        gurobi_status=gurobi_status,
        gurobi_status_name=gurobi_status_name,
        gurobi_obj_value=gurobi_obj_value,
        verifier_report=verifier_report,
        trace=trace,
    )



def create_client(cfg: PipelineConfig, test_flag: bool = False, type: str = "default") -> OpenAI:
    """
    创建OpenAI客户端，用于兼容本地的网络环境设置，type只有default和其他两种选择。
    """
    print("Creating OpenAI client with model:", cfg.model_name)

    if type != "default":
        import httpx
        transport = httpx.HTTPTransport(
        proxy="http://127.0.0.1:7890",
            verify=False  # 关闭SSL验证
        )

        http_client = httpx.Client(
            transport=transport,
            timeout=60
        )

        # LLM连通性测试
        client_test = OpenAI(http_client=http_client)
    else:
        client_test = OpenAI()

    # 发送Hello并接受并打印响应
    if test_flag:
        try:
            response = client_test.chat.completions.create(
                model=cfg.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                temperature=cfg.temperature,
                max_tokens=5,
            )
            print("LLM连通性测试响应:", response.choices[0].message.content)
        except Exception as e:
            print("LLM连通性测试失败:", str(e))
            raise e

    return client_test