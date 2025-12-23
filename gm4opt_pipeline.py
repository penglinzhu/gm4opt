# gm4opt_pipeline.py
# 统一流水线：NL → (dynamic system prompt templating) → IR(JSON) → ModelIR （→ GraphIR → Difficulty） → Gurobi

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI
from gurobipy import GRB

from gm4opt_ir import ModelIR, ir_to_gurobi
from gm4opt_nl2ir import (
    build_system_prompt,
    build_user_prompt,
    extract_json_from_text,
    json_to_model_ir,
)
from gm4opt_graph import build_graph_from_ir
from gm4opt_difficulty import DifficultyEstimate, attach_difficulty


@dataclass
class PipelineConfig:
    """
    GM4OPT 主流水线配置。

    - model_name: 使用的 LLM 模型名称
    - temperature: 采样温度
    - enable_dynamic_prompt: 是否启用动态 system prompt templating
    - enable_graph: 是否根据 IR 构建 GraphIR
    - enable_difficulty: 是否在 GraphIR 基础上计算难度估计
    - solve_with_gurobi: 是否在得到 IR 后直接调用 Gurobi 求解
    """
    model_name: str = "gpt-4o"
    temperature: float = 0.0

    enable_dynamic_prompt: bool = True
    enable_graph: bool = True
    enable_difficulty: bool = True
    solve_with_gurobi: bool = False


@dataclass
class PipelineResult:
    """
    GM4OPT 主流水线运行结果。
    """
    raw_llm_text: str              # 模型原始输出（字符串）
    ir_dict: Dict[str, Any]        # 解析后的 JSON 字典
    ir: ModelIR                    # ModelIR 对象
    graph_built: bool              # 是否构建了 GraphIR
    difficulty: Optional[DifficultyEstimate]
    gurobi_status: Optional[int]
    gurobi_obj_value: Optional[float]


def run_gm4opt_pipeline(
    question_text: str,
    client: Optional[OpenAI] = None,
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """
    GM4OPT 主入口：给定一个自然语言优化问题描述，执行：

      question_text
        → dynamic system prompt templating
        → NL→IR LLM 调用
        → JSON dict
        → ModelIR
        → (可选 GraphIR + Difficulty)
        → Gurobi 求解

    返回 PipelineResult。
    """
    if client is None:
        client = OpenAI()
    if config is None:
        config = PipelineConfig()

    #1) 构造动态 system prompt
    system_prompt = build_system_prompt(
        question_text=question_text,
        enable_dynamic=config.enable_dynamic_prompt,
        difficulty_hint=None,
    )

    #2) 构造 user prompt
    user_prompt = build_user_prompt(question_text)

    #3) messages
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    #4) 调用 LLM
    completion = client.chat.completions.create(
        model=config.model_name,
        messages=messages,
        temperature=config.temperature,
    )
    raw_llm_text: str = completion.choices[0].message.content or ""

    #5) 抽取 JSON
    data = extract_json_from_text(raw_llm_text)

    if "meta" not in data or not isinstance(data["meta"], dict):
        data["meta"] = {}
    if not data["meta"].get("problem_id"):
        data["meta"]["problem_id"] = "gm4opt_instance"

    #6) ModelIR
    ir = json_to_model_ir(data)

    #7) 可选：构建 GraphIR + 难度估计
    graph_built = False
    difficulty: Optional[DifficultyEstimate] = None

    if config.enable_graph:
        ir.graph = build_graph_from_ir(ir)
        graph_built = True

    if config.enable_difficulty:
        difficulty = attach_difficulty(ir)

    #8) Gurobi 求解
    gurobi_status: Optional[int] = None
    gurobi_obj_value: Optional[float] = None

    if config.solve_with_gurobi:
        model = ir_to_gurobi(ir)
        model.setParam("TimeLimit", 60.0)
        model.optimize()
        gurobi_status = model.Status
        if model.Status == GRB.OPTIMAL:
            gurobi_obj_value = float(model.ObjVal)

    return PipelineResult(
        raw_llm_text=raw_llm_text,
        ir_dict=data,
        ir=ir,
        graph_built=graph_built,
        difficulty=difficulty,
        gurobi_status=gurobi_status,
        gurobi_obj_value=gurobi_obj_value,
    )
