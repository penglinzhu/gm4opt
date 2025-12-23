# run_nl2ir_demo.py
# 用 GM4OPT 统一 pipeline 跑通一条题目（单样例），并打印：raw 输出、graph/difficulty、Gurobi 解


from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Dict

from openai import OpenAI

from gm4opt_pipeline import PipelineConfig, run_gm4opt_pipeline


def _build_pipeline_config(**kwargs) -> PipelineConfig:
    """
    根据 PipelineConfig 的真实字段，自动筛选 kwargs 并构造 config。
    """
    if not is_dataclass(PipelineConfig):
        return PipelineConfig(**kwargs) 

    field_names = set(PipelineConfig.__dataclass_fields__.keys()) 
    filtered: Dict[str, Any] = {k: v for k, v in kwargs.items() if k in field_names}
    return PipelineConfig(**filtered)  


def main():
    # 可以换成任意题目
    QUESTION_TEXT = """
Welcome to the fictional city of Aquaville, where the city's administration is working on a groundbreaking project to maximize the efficiency of its new water distribution network. This network is designed to ensure that every corner of the city has access to water, especially in times of need. The system is made up of 9 pivotal stations - including the main water source and the central distribution hub - connected by a series of pipelines with varying capacities.\n\nHere's a closer look at the capacities (in thousands of gallons per hour) that each pipeline can handle, showcasing the maximum flow from one station to another across the city:\n\n- From Station 0 (Source): Water can be directed to Station 1 (2 gallons), Station 2 (8 gallons), Station 3 (17 gallons), Station 4 (16 gallons), Station 5 (20 gallons), Station 6 (15 gallons), Station 7 (12 gallons), and Station 8 (9 gallons).\n- From Station 1: Water can flow to Station 0 (4 gallons), Station 2 (11 gallons), Station 3 (7 gallons), Station 4 (3 gallons), Station 5 (19 gallons), Station 6 (10 gallons), Station 7 (19 gallons), and Station 8 (8 gallons).\n- From Station 2: Water can move to Station 0 (14 gallons), Station 1 (7 gallons), Station 3 (7 gallons), Station 4 (13 gallons), Station 5 (19 gallons), Station 6 (18 gallons), Station 7 (20 gallons), and Station 8 (5 gallons).\n- From Station 3: Water can be sent to Station 0 (17 gallons), Station 1 (17 gallons), Station 2 (7 gallons), Station 4 (11 gallons), Station 5 (7 gallons), Station 6 (17 gallons), Station 7 (0 gallons), and Station 8 (10 gallons).\n- From Station 4: Water can be distributed to Station 0 (20 gallons), Station 1 (10 gallons), Station 2 (2 gallons), Station 3 (18 gallons), Station 5 (15 gallons), Station 6 (8 gallons), Station 7 (6 gallons), and Station 8 (18 gallons).\n- From Station 5: Water can be channeled to Station 0 (4 gallons), Station 1 (9 gallons), Station 2 (12 gallons), Station 3 (1 gallon), Station 4 (17 gallons), Station 6 (0 gallons), Station 7 (13 gallons), and Station 8 (18 gallons).\n- From Station 6: Water can flow to Station 0 (12 gallons), Station 1 (15 gallons), Station 2 (12 gallons), Station 3 (19 gallons), Station 4 (18 gallons), Station 5 (19 gallons), Station 7 (6 gallons), and Station 8 (17 gallons).\n- From Station 7: Water can be directed to Station 0 (14 gallons), Station 1 (1 gallon), Station 2 (17 gallons), Station 3 (20 gallons), Station 4 (11 gallons), Station 5 (20 gallons), Station 6 (6 gallons), and Station 8 (4 gallons).\n- From Station 8 (Destination): Water can be sent back to Station 0 (19 gallons), Station 1 (3 gallons), Station 2 (19 gallons), Station 3 (12 gallons), Station 4 (8 gallons), Station 5 (18 gallons), Station 6 (17 gallons), and Station 7 (13 gallons).\n\nIn this intricate network, a capacity of \"0\" implies that there is no direct connection for water flow between those two specific stations.\n\nThe critical mission is to determine the maximum quantity of water that can be efficiently distributed from Station 0, the primary source, to Station 8, the central distribution hub, per hour. The flow through each pipeline must not exceed its maximum specified capacity.\n\nWhat is the optimal amount of water (in thousands of gallons per hour) that can be transported from the source to the distribution hub, ensuring the network operates at its maximum potential?
""".strip()

    client = OpenAI()  # 从环境变量读取 OPENAI_API_KEY

    # 在这里改开关做消融实验
    cfg = _build_pipeline_config(
        model_name="gpt-4o",
        temperature=0.0,

        enable_dynamic_prompt=True,
        enable_graph=True,
        enable_difficulty=True,
        solve_with_gurobi=True,
    )

    res = run_gm4opt_pipeline(
        question_text=QUESTION_TEXT,
        client=client,
        config=cfg,
    )

    # ===== 输出：LLM 原始输出 =====
    raw_text = getattr(res, "raw_llm_text", None)
    if raw_text is not None:
        print("=== Raw LLM output ===")
        print(raw_text)
        print("=== End of raw LLM output ===\n")

    # ===== 输出：Graph（如果启用）=====
    print("Graph built:", getattr(res, "graph_built", None))

    # ===== 输出：Difficulty（如果启用）=====
    diff = getattr(res, "difficulty", None)
    if diff is not None:
        print(f"\nDifficulty: score={diff.score:.3f}, level={diff.level}")
        print("Difficulty features:")
        for k, v in diff.features.items():
            print(f"  {k}: {v}")

    # ===== 输出：Gurobi 求解结果 =====
    status = getattr(res, "gurobi_status", None)
    if status is not None:
        print("\n=== Gurobi ===")
        print("Status code:", status)
        obj_val = getattr(res, "gurobi_obj_value", None)
        if obj_val is not None:
            print("Objective value:", obj_val)
        else:
            print("No optimal objective value obtained.")

    # ===== 输出：Graph meta（如果构建了图）=====
    ir = getattr(res, "ir", None)
    if ir is not None and getattr(ir, "graph", None) is not None:
        gmeta = ir.graph.meta or {}
        print("\n=== Graph meta ===")
        print("Graph stats:", gmeta.get("stats", {}))
        print("Graph checks:")
        for issue in gmeta.get("checks", {}).get("issues", []):
            print("  -", issue)


if __name__ == "__main__":
    main()
