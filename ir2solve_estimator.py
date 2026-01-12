# ir2solve_estimator.py
# 基于 ModelIR + GraphIR 的简单 difficulty-aware 启发式（以图结构为主）

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ir2solve_ir import ModelIR

import gurobipy as gp

import json

import re

# ========= Prompts =========

BASE_SYSTEM_PROMPT = """
You are an expert in mathematical modeling and optimization.
Your task is to:
1. Read a natural language description of a linear/integer optimization problem
2. Read a JSON-formatted mathematical model built for this problem
3. [Optional]Read difficulty features of this model in JSON format

Based on these inputs, you must:
- Provide a confidence score between 0 and 10 for the model's correctness, bigger is more confident
- If the confidence score is less than 5, provide a corrected JSON-formatted mathematical model

IMPORTANT: You MUST output a JSON object with the following structure:
{
  "confidence_score": <number between 0 and 10>,
  "corrected_model": <JSON object following the schema below, or null if confidence_score >= 5>
  "reasoning": "string explaining the confidence score and correction"
}

To determine the confidence_score, evaluate the model based on the following criteria:

1. Verification of Constraints Accuracy: Evaluate whether the constraints in the model align with the content of the problem description.
2. Verification of Parameter Correctness: Assess whether each coefficient in the objective function and constraints has a direct or implicit source.
3. Verification of Objective Function Correctness: Determine whether the objective function in the model reflects the optimization goal stated in the problem.
"""

SCHEMA_FOR_MODEL = r"""
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
      "sense": ">=", "<=", or "=",
      "rhs": <float>,
      "expression": "string such as '3x1-4x2+5x3'"
    }
  ]
}
"""
SCHEMA_FOR_FEATURES = r"""
IF DIFFICULTY FEATURES ARE GIVEN, DIFFICULTY FEATURES SCHEMA:
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


def build_check_message(question_txt: str, features: Dict[str, Any] = None, gurobi_model: gp.model = None) -> List[Dict[str, str]]:
    """
    生成评估与自检的LLM message
    """

    system_prompt = BASE_SYSTEM_PROMPT
    
    # 提取模型信息
    model_info = None
    if gurobi_model is not None:
        try:
            objective, variables, constraints, error = extract_model_info(gurobi_model)
            if error is not None:
                model_info = {"error": error}
        except Exception as e:
            model_info = {"error": f"model_extraction_exception: {str(e)}"}
            raise ValueError(f"Model extraction failed: {str(e)}")

        model_info = {
            "objective": objective,
            "variables": variables,
            "constraints": constraints
        }
    
    model_info_str = json.dumps(model_info, ensure_ascii=False, indent=2) if model_info is not None else "null"
    # print("Model Info for Difficulty Check:", model_info_str)
    features_str = json.dumps(features, ensure_ascii=False, indent=2) if features is not None else "{}"
    
    user_prompt = f"""{SCHEMA_FOR_MODEL}{SCHEMA_FOR_FEATURES}

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

def extract_model_info(model: gp.Model) -> tuple:
    """
    提取模型的完整信息
    
    Args:
        model: Gurobi模型
    
    Returns:
        tuple: (objective_dict, variables_list, constraints_list, error)
    """
    try:
        # 1. 提取目标函数
        objective = {
            "direction": "minimize" if model.ModelSense == 1 else "maximize",
            "expression": "",
            "constant": 0
        }
        
        obj = model.getObjective()
        if obj:
            # 获取目标函数的线性表达式
            obj_expr = str(obj)
            objective["expression"] = obj_expr
        
        # 2. 提取变量
        variables = []
        for var in model.getVars():
            # 确定变量类型字符
            vtype = var.VType
            if vtype == gp.GRB.BINARY:
                type_char = "B"
            elif vtype == gp.GRB.INTEGER:
                type_char = "I"
            else:
                type_char = "C"
            
            # 处理边界
            lb = var.LB
            ub = var.UB
            
            if lb == -float('inf') or lb <= -1e30:
                lb_str = "-inf"
            else:
                lb_str = float(lb)
            
            if ub == float('inf') or ub >= 1e30:
                ub_str = "inf"
            else:
                ub_str = float(ub)
            
            variables.append({
                "name": var.VarName,
                "type": type_char,
                "lower_bound": lb_str,
                "upper_bound": ub_str
            })
        
        # 3. 提取约束
        constraints = []
        for constr in model.getConstrs():
            try:
                # 获取约束行
                row = model.getRow(constr)
                
                # 构建左边表达式字符串
                expr_parts = []
                for i in range(row.size()):
                    var = row.getVar(i)
                    coeff = row.getCoeff(i)
                    
                    if coeff == 0:
                        continue
                    
                    coeff_val = float(coeff)
                    var_name = var.VarName
                    
                    # 格式化项
                    if coeff_val == 1:
                        term = var_name
                    elif coeff_val == -1:
                        term = f"-{var_name}"
                    else:
                        term = f"{coeff_val}{var_name}"
                    
                    expr_parts.append(term)
                
                # 构建完整表达式
                if expr_parts:
                    # 处理第一项
                    left_expr = expr_parts[0]
                    
                    # 处理后续项
                    for term in expr_parts[1:]:
                        if term.startswith('-'):
                            left_expr += f" {term}"
                        else:
                            left_expr += f" + {term}"
                else:
                    left_expr = "0"
                
                # 获取约束类型
                sense_map = {
                    '<': '<=',
                    '>': '>=',
                    '=': '='
                }
                sense = sense_map.get(constr.Sense, '=')
                
                # 获取右边值
                rhs = float(constr.RHS)
                
                constraints.append({
                    "name": constr.ConstrName if constr.ConstrName else f"constr_{constr.index}",
                    "sense": sense,
                    "expression": f"{left_expr} {sense} {rhs}",
                    "rhs": rhs
                })
                
            except Exception as e:
                # 如果API调用失败，使用简单方法
                constraints.append({
                    "name": constr.ConstrName if constr.ConstrName else f"constr_{constr.index}",
                    "sense": "=",
                    "expression": f"constraint_{constr.index}",
                    "rhs": 0
                })
        
        return objective, variables, constraints, None
        
    except Exception as e:
        return None, None, None, f"extract_model_error: {str(e)}"

# ======= 从dict重建模型 =======

# 将dict格式的模型重建为gurobipy.Model
def dict_rebuild_model(model_dict: Dict) -> gp.Model:
    """
    将dict格式的模型重建为gurobipy.Model
    
    Args:
        model_dict: 包含模型信息的字典，格式为：
            {
                "objective": {
                    "direction": "minimize"/"maximize",
                    "expression": "表达式字符串",
                    "constant": 常数项
                },
                "variables": [
                    {
                        "name": "变量名",
                        "type": "B"/"I"/"C",
                        "lower_bound": 下界,
                        "upper_bound": 上界
                    },
                    ...
                ],
                "constraints": [
                    {
                        "name": "约束名",
                        "sense": "="/">="/"<=",
                        "rhs": 右边常数,
                        "expression": "表达式字符串"
                    },
                    ...
                ]
            }
    
    Returns:
        gurobipy.Model: 重建的Gurobi模型
    """
    try:
        # 验证输入
        if not isinstance(model_dict, dict):
            raise ValueError("输入必须是字典类型")
        
        # 创建新模型
        model = gp.Model("rebuilt_from_dict")
        
        # 1. 添加变量
        var_dict = {}
        for var_info in model_dict.get("variables", []):
            var_name = var_info.get("name", "")
            if not var_name:
                continue
            
            # 确定变量类型
            type_map = {"B": gp.GRB.BINARY, "I": gp.GRB.INTEGER, "C": gp.GRB.CONTINUOUS}
            var_type = var_info.get("type", "C").upper()
            vtype = type_map.get(var_type, gp.GRB.CONTINUOUS)
            
            # 处理边界值
            lb = var_info.get("lower_bound", 0.0)
            ub = var_info.get("upper_bound", 1.0 if var_type == "B" else gp.GRB.INFINITY)
            
            # 处理字符串形式的无限值
            if isinstance(lb, str) and lb.lower() in ["-inf", "-infinity"]:
                lb = -gp.GRB.INFINITY
            if isinstance(ub, str) and ub.lower() in ["inf", "infinity"]:
                ub = gp.GRB.INFINITY
            
            # 创建变量
            try:
                var = model.addVar(
                    lb=float(lb),
                    ub=float(ub),
                    vtype=vtype,
                    name=str(var_name)
                )
                var_dict[var_name] = var
            except Exception as e:
                raise ValueError(f"添加变量 '{var_name}' 失败: {e}")
        
        # 更新模型以包含变量
        model.update()
        
        # 2. 设置目标函数
        obj_info = model_dict.get("objective", {})
        if "expression" in obj_info:
            expr_str = obj_info.get("expression", "")
            if expr_str:
                # 解析目标函数表达式
                expr = _parse_linear_expression(expr_str, var_dict)
                
                # 添加常数项
                constant = obj_info.get("constant", 0)
                if constant != 0:
                    expr += float(constant)
                
                # 设置目标方向
                direction = obj_info.get("direction", "minimize")
                if direction == "minimize":
                    model.setObjective(expr, gp.GRB.MINIMIZE)
                else:
                    model.setObjective(expr, gp.GRB.MAXIMIZE)
        
        # 3. 添加约束
        for constr_info in model_dict.get("constraints", []):
            constr_name = constr_info.get("name", "")
            sense = constr_info.get("sense", "=")
            rhs = constr_info.get("rhs", 0)
            expr_str = constr_info.get("expression", "")
            
            if not expr_str:
                continue
            
            # 解析约束表达式
            expr = _parse_linear_expression(expr_str, var_dict)
            
            # 添加约束（使用正确的语法）
            if sense == "<=":
                model.addConstr(expr <= rhs, name=constr_name)
            elif sense == ">=":
                model.addConstr(expr >= rhs, name=constr_name)
            else:  # "=" 或其他
                model.addConstr(expr == rhs, name=constr_name)
        
        # 4. 更新模型
        model.update()
        
        return model
        
    except Exception as e:
        raise ValueError(f"模型重建失败: {str(e)}")

# 解析线性表达式字符串为Gurobi线性表达式
def _parse_linear_expression(expr_str: str, var_dict: Dict[str, gp.Var]) -> gp.LinExpr:
    """
    解析线性表达式字符串为Gurobi线性表达式
    
    支持格式: 
    - "9 assign[I,A] + 4 assign[I,B]"
    - "assign[I,A] + assign[II,A]" (系数为1)
    - "-3 assign[I,A] + 2 assign[I,B]"
    - "assign[I,A] + assign[II,A] = 1" (会忽略=和右边部分)
    """
    # 初始化线性表达式
    expr = gp.LinExpr(0)
    
    # 如果表达式为空，返回空表达式
    if not expr_str or not expr_str.strip():
        return expr
    
    # 清理表达式：移除所有空格
    expr_str = expr_str.replace(" ", "")
    
    # 如果表达式包含比较运算符，只取左边部分
    for op in ["<=", ">=", "==", "=", "<", ">"]:
        if op in expr_str:
            expr_str = expr_str.split(op)[0]
            break
    
    # 如果表达式以"+"开头，去掉它
    if expr_str.startswith("+"):
        expr_str = expr_str[1:]
    
    # 使用正则表达式匹配每一项
    # 模式解释:
    # ([+-]?\d*\.?\d*) - 可选的符号和系数（整数或小数）
    # ([a-zA-Z_][a-zA-Z0-9_\[\],]*) - 变量名
    pattern = r'([+-]?\d*\.?\d*)([a-zA-Z_][a-zA-Z0-9_\[\],]*)'
    
    # 查找所有匹配项
    for match in re.finditer(pattern, expr_str):
        coeff_str = match.group(1)
        var_name = match.group(2)
        
        # 处理系数
        if not coeff_str or coeff_str in ["+", "-"]:
            # 没有显式系数或只有符号
            coeff = 1.0 if coeff_str != "-" else -1.0
        else:
            try:
                coeff = float(coeff_str)
            except ValueError:
                coeff = 1.0
        
        # 查找变量并添加到表达式
        if var_name in var_dict:
            expr.addTerms(coeff, var_dict[var_name])
        else:
            # 如果变量名不在字典中，尝试清理变量名
            clean_var_name = var_name.strip()
            if clean_var_name in var_dict:
                expr.addTerms(coeff, var_dict[clean_var_name])
            else:
                # 如果还是找不到，抛出错误
                raise ValueError(f"未找到变量: '{var_name}'")
    
    return expr


# 整合函数：从字符串到模型
def load_model_from_dict_str(dict_str: str) -> gp.Model:
    """
    从字典字符串加载模型
    Args:
        dict_str: 字典格式的模型字符串
    Returns:
        gurobipy.Model: 重建的模型
    """
    try:
        # 尝试解析为JSON
        import ast
        model_dict = ast.literal_eval(dict_str)
        return dict_rebuild_model(model_dict)
    except Exception as e:
        raise ValueError(f"从字典字符串加载模型失败: {str(e)}")