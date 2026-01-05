# gm4opt_difficulty.py
# 基于 ModelIR + GraphIR 的简单 difficulty-aware 启发式（以图结构为主）

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from gm4opt_ir import ModelIR

import gurobipy as gp

import json

# ========= Prompts =========

BASE_SYSTEM_PROMPT = """
You are an expert in mathematical modeling and optimization.
Your task is to:
1. Read a natural language description of a linear/integer optimization problem
2. Read a JSON-formatted mathematical model built for this problem
3. Read difficulty features of this model in JSON format

Based on these inputs, you must:
- Provide a confidence score between 0 and 10 for the model's correctness
- If the confidence score is less than 5, provide a corrected JSON-formatted mathematical model

IMPORTANT: You MUST output a JSON object with the following structure:
{
  "confidence_score": <number between 0 and 10>,
  "corrected_model": <JSON object following the schema below, or null if confidence_score >= 5>
}
"""

SCHEMA_AND_INSTRUCTIONS = r"""
SCHEMA FOR THE MATHEMATICAL MODEL:
The JSON-formatted mathematical model follows this structure. 
If you need to correct the model (confidence_score < 5), you MUST produce valid JSON in this exact format:

{
  "objective": {
    "direction": "minimize" or "maximize",
    "expression": "string such as '3x1-4x2+5x3'",
    "constant": <float>
  },
  "variables": [
    {
      "name": "string",
      "type": "C" (continuous), "I" (integer), or "B" (binary),
      "lower_bound": <float> or "-inf",
      "upper_bound": <float> or "inf"
    }
  ],
  "constraints": [
    {
      "name": "string",
      "sense": ">=", "<=", or "==",
      "rhs": <float>,
      "expression": "string such as '3x1-4x2+5x3'"
    }
  ]
}

DIFFICULTY FEATURES SCHEMA:
The difficulty features are provided in this format:
{
  "n_sets": <int>,  # Number of set definitions
  "n_params": <int>,  # Number of parameter definitions
  "n_vars": <int>,  # Number of decision variables
  "n_constraints": <int>,  # Number of constraints
  "n_binary_vars": <int>,  # Number of binary (0-1) variables
  "n_integer_vars": <int>,  # Number of integer variables (non-binary)
  "n_continuous_vars": <int>,  # Number of continuous variables
  "n_2d_vars": <int>,  # Number of variables with two-dimensional indexing
  "n_2d_params": <int>,  # Number of parameters with two-dimensional indexing
  "n_graph_nodes": <int>,  # Total nodes in graph representation
  "n_graph_edges": <int>,  # Total edges in graph representation
  "n_connected_components": <int>,  # Number of connected components in graph
  "n_orphan_vars": <int>,  # Variables not appearing in objective/constraints
  "n_unused_params": <int>,  # Parameters not appearing in objective/constraints
  "n_unused_sets": <int>  # Sets not connected to any parameter/variable
}
"""

@dataclass
class DifficultyEstimate:
    """
    对单个 ModelIR 的难度估计结果。

    - score: 一个连续的难度分数（越大越难）
    - level: "easy" / "medium" / "hard" 等粗粒度标签
    - features: 用于估计的特征字典
    """
    score: float
    level: str
    features: Dict[str, Any]


def build_check_message(question_txt: str, features: Dict[str, Any], gurobi_model: gp.model) -> List[Dict[str, str]]:
    """
    生成评估与自检的LLM message
    """

    system_prompt = BASE_SYSTEM_PROMPT
    
    # 提取模型信息
    model_info = None
    if gurobi_model is not None:
        model_info = {
            "objective": extract_full_objective(gurobi_model),
            "variables": extract_all_variables(gurobi_model),
            "constraints": extract_all_constraints(gurobi_model)
        }
    
    model_info_str = json.dumps(model_info, ensure_ascii=False, indent=2) if model_info is not None else "null"
    features_str = json.dumps(features, ensure_ascii=False, indent=2) if features is not None else "{}"
    
    user_prompt = f"""{SCHEMA_AND_INSTRUCTIONS}

    TASK EXECUTION:
    1. PROBLEM DESCRIPTION:
    {question_txt}

    2. PROVIDED MATHEMATICAL MODEL:
    {model_info_str}

    3. DIFFICULTY FEATURES:
    {features_str}

    YOUR OUTPUT MUST BE A VALID JSON OBJECT with 'confidence_score' and 'corrected_model' keys.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages

def estimate_difficulty(ir: ModelIR) -> DifficultyEstimate:
    """
    根据 ModelIR（尤其是 GraphIR）、gurobi求解结果共同估计一个难度分数/建模置信度。
    """

    #1. IR基础计数
    n_sets = len(ir.sets)
    n_params = len(ir.params)
    n_vars = len(ir.vars)
    n_constraints = len(ir.constraints)

    # 各类变量数量
    n_binary_vars = sum(1 for v in ir.vars if v.vartype == "binary")
    n_integer_vars = sum(1 for v in ir.vars if v.vartype == "integer")
    n_continuous_vars = n_vars - n_binary_vars - n_integer_vars

    # 变量/参数维度信息
    n_2d_vars = sum(1 for v in ir.vars if len(v.indices) == 2)
    n_2d_params = sum(1 for p in ir.params if len(p.indices) == 2)

    #2. GraphIR相关特征
    n_graph_nodes = 0
    n_graph_edges = 0
    n_connected_components = 0
    n_orphan_vars = 0
    n_unused_params = 0
    n_unused_sets = 0

    if ir.graph is not None:
        stats = ir.graph.meta.get("stats", {}) if ir.graph.meta else {}
        n_graph_nodes = stats.get("n_nodes", len(ir.graph.nodes))
        n_graph_edges = stats.get("n_edges", len(ir.graph.edges))
        n_connected_components = stats.get("n_connected_components", 0)

        checks = ir.graph.meta.get("checks", {}) if ir.graph.meta else {}
        issues = checks.get("issues", [])

        for iss in issues:
            kind = iss.get("kind", "")
            nodes = iss.get("nodes", []) or []
            if kind == "orphan_var":
                n_orphan_vars += len(nodes)
            elif kind == "unused_param":
                n_unused_params += len(nodes)
            elif kind == "unused_set":
                n_unused_sets += len(nodes)
            elif kind == "disconnected_component":
                pass

    #3. 汇总特征
    features: Dict[str, Any] = {
        "n_sets": n_sets,
        "n_params": n_params,
        "n_vars": n_vars,
        "n_constraints": n_constraints,
        "n_binary_vars": n_binary_vars,
        "n_integer_vars": n_integer_vars,
        "n_continuous_vars": n_continuous_vars,
        "n_2d_vars": n_2d_vars,
        "n_2d_params": n_2d_params,
        "n_graph_nodes": n_graph_nodes,
        "n_graph_edges": n_graph_edges,
        "n_connected_components": n_connected_components,
        "n_orphan_vars": n_orphan_vars,
        "n_unused_params": n_unused_params,
        "n_unused_sets": n_unused_sets,
    }

    #4. 线性打分
    score = 0.0

    # 基本结构复杂度（规模越大越难）
    score += 0.5 * n_sets
    score += 0.8 * n_params
    score += 1.0 * n_vars
    score += 0.7 * n_constraints

    # 变量类型影响：整数/二进制更难
    score += 2.0 * n_integer_vars
    score += 3.0 * n_binary_vars

    # 高维结构带来的额外复杂度
    score += 1.0 * n_2d_vars
    score += 0.5 * n_2d_params

    # 图结构规模：轻微加权
    score += 0.05 * n_graph_nodes
    score += 0.02 * n_graph_edges

    # 图的拓扑复杂度：多个连通分量通常表示模型结构较碎、依赖关系更复杂
    if n_connected_components > 1:
        score += 2.0 * (n_connected_components - 1)

    # 建模质量问题：orphan_var / unused_param / unused_set
    score += 1.0 * n_orphan_vars
    score += 0.5 * n_unused_params
    score += 0.3 * n_unused_sets

    #5. 粗粒度difficulty level
    # 阈值目前是随便设置的
    if score < 10:
        level = "easy"
    elif score < 25:
        level = "medium"
    else:
        level = "hard"
    
    # 把结果挂到 graph.meta 里，方便后续使用
    if ir.graph is not None:
        if ir.graph.meta is None:
            ir.graph.meta = {}
        ir.graph.meta["difficulty"] = {
            "score": score,
            "level": level,
            "features": features,
        }

    return DifficultyEstimate(score=score, level=level, features=features)



# =======解析model信息=======

def extract_full_objective(model):
    """提取完整的目标函数表达式"""
    obj_info = {
        "direction": "minimize" if model.ModelSense == 1 else "maximize",
        "expression": "",  # 字符串形式
        "constant": 0,
    }
    
    try:
        obj = model.getObjective()
        
        # 收集所有项
        terms = []
        first_term = True
        
        for var in model.getVars():
            coeff = obj.getCoeff(var)
            if coeff != 0:
                coeff_val = float(coeff)
                
                if first_term:
                    # 第一项处理
                    if coeff_val == 1:
                        terms.append(f"{var.VarName}")
                    elif coeff_val == -1:
                        terms.append(f"-{var.VarName}")
                    else:
                        terms.append(f"{coeff_val}{var.VarName}")
                    first_term = False
                else:
                    # 后续项处理
                    if coeff_val > 0:
                        if coeff_val == 1:
                            terms.append(f"+ {var.VarName}")
                        else:
                            terms.append(f"+ {coeff_val}{var.VarName}")
                    else:  # coeff_val < 0
                        if coeff_val == -1:
                            terms.append(f"- {var.VarName}")
                        else:
                            terms.append(f"- {abs(coeff_val)}{var.VarName}")
        
        # 处理常数项
        constant = obj.getConstant()
        if constant != 0:
            obj_info["constant"] = float(constant)
            const_val = float(constant)
            
            if first_term:
                terms.append(f"{const_val}")
                first_term = False
            else:
                if const_val > 0:
                    terms.append(f"+ {const_val}")
                else:
                    terms.append(f"- {abs(const_val)}")
        
        # 构建表达式字符串
        if terms:
            # 移除所有空格并合并
            expression = "".join(terms).replace(" ", "")
            
            # 进一步清理：处理像"3.0x1"这样的格式
            import re
            expression = re.sub(r'(\d+)\.0+(\D|$)', r'\1\2', expression)
            
            obj_info["expression"] = expression
              
    except Exception as e:
        obj_info["error"] = f"extract_obj_error: {str(e)}"
    
    return obj_info


def extract_all_variables(model):
    """提取所有变量的完整信息"""
    variables = []
    
    try:
        for var in model.getVars():
            var_info = {
                "name": var.VarName,
                "type": var.VType,
                "lower_bound": float(var.LB) if var.LB != -float('inf') else "-inf",
                "upper_bound": float(var.UB) if var.UB != float('inf') else "inf",
            }
            variables.append(var_info)
    
    except Exception as e:
        return {"error": f"extract_var_error: {str(e)}"}
    
    return variables


def extract_all_constraints(model):
    """提取所有约束的完整信息"""
    constraints = []
    
    def get_constraint_sense(constr):
        """获取约束类型"""
        sense_map = {0: "<=", 1: ">=", 2: "=="}
        return sense_map.get(constr.Sense, "unknown")
    
    def format_coeff(value):
        """格式化系数"""
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    
    try:
        for constr in model.getConstrs():
            constr_info = {
                "name": constr.ConstrName,
                "sense": get_constraint_sense(constr),
                "rhs": float(constr.RHS),
                "expression": ""  # 字符串形式的表达式
            }
            
            try:
                row = model.getRow(constr)
                expression_parts = []
                first = True
                
                # 处理左边表达式
                for i in range(row.size()):
                    var = row.getVar(i)
                    coeff = row.getCoeff(i)
                    
                    if coeff != 0:
                        coeff_val = float(coeff)
                        formatted_coeff = format_coeff(abs(coeff_val))
                        
                        if first:
                            # 第一项
                            if coeff_val > 0:
                                if coeff_val == 1:
                                    expression_parts.append(var.VarName)
                                else:
                                    expression_parts.append(f"{formatted_coeff}{var.VarName}")
                            else:  # coeff_val < 0
                                if coeff_val == -1:
                                    expression_parts.append(f"-{var.VarName}")
                                else:
                                    expression_parts.append(f"-{formatted_coeff}{var.VarName}")
                            first = False
                        else:
                            # 后续项
                            if coeff_val > 0:
                                if coeff_val == 1:
                                    expression_parts.append(f"+{var.VarName}")
                                else:
                                    expression_parts.append(f"+{formatted_coeff}{var.VarName}")
                            else:  # coeff_val < 0
                                if coeff_val == -1:
                                    expression_parts.append(f"-{var.VarName}")
                                else:
                                    expression_parts.append(f"-{formatted_coeff}{var.VarName}")
                
                # 构建左边表达式字符串
                if expression_parts:
                    left_expr = "".join(expression_parts)
                else:
                    left_expr = "0"
                
                # 构建完整约束表达式
                constr_info["expression"] = f"{left_expr} {constr_info['sense']} {format_coeff(constr_info['rhs'])}"
                
            except Exception as e:
                constr_info["error"] = f"提取约束表达式失败: {str(e)}"
            
            constraints.append(constr_info)
    
    except Exception as e:
        return {"error": f"约束提取失败: {str(e)}"}
    
    return constraints

