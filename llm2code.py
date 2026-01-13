# llm2code.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ir2solve_ir import ModelIR

import gurobipy as gp

import json

import warnings

import ast

BASE_SYSTEM_PROMPT = """
You are an expert in mathematical modeling and optimization.
Your task is to:
1. Read a natural language description of a linear/integer optimization problem.
2. Read the provided JSON immediate representation (ModelIR) of the modeling process of the problem.
3. Generate Python code that uses the Gurobi library to create the optimization model based on the ModelIR.

IMPORTANT: You MUST generate a Python function named `build_gurobi_model()` that:
1. Takes no parameters
2. Creates a Gurobi model using the provided information
3. Returns the created model

The generated Python code should follow these guidelines:
1. Import necessary libraries: import gurobipy as gp
2. Create the model: model = gp.Model("LP_Model")
3. Add variables according to variable information:
   - Continuous variables: model.addVar(lb=..., ub=..., vtype=gp.CONTINUOUS, name=...)
   - Integer variables: model.addVar(vtype=gp.INTEGER, name=...)
   - Binary variables: model.addVar(vtype=gp.BINARY, name=...)
4. Add constraints: model.addConstr(...)
5. Set objective function: model.setObjective(...)
6. Define a function `build_gurobi_model()` that creates and returns the model
7. At the end, call the function to build the model

Please output the complete Python code directly without any additional explanations.

EXAMPLE FORMAT:
```python
import gurobipy as gp
from gurobipy import GRB

def build_gurobi_model():
    # Create a new model
    model = gp.Model("LP_Model")
    
    # Your modeling code here...
    
    return model

# Build the model
model = build_gurobi_model()
""".strip()

SCHEMA_FOR_IR = r"""
SCHEMA FOR THE INTERMEDIATE REPRESENTATION (IR):
The IR is a structured representation of the problem, typically in JSON format.
It is not always correct. It includes the following components:

1) sets
A list of sets. Each set is:
{
  "name": "string",
  "elements": ["string", ...],   // IMPORTANT: every element MUST be a STRING (even if it looks like a number)
  "description": "string or null"
}

2) params
A list of parameters. Each param is:
{
  "name": "string",
  "indices": [] | ["SetName"] | ["SetName1","SetName2"],  // 0D/1D/2D only
  "values": number | {key:number,...} | {i:{j:number,...},...},   // recommended canonical forms below
  "description": "string or null"
}

Canonical forms for values:
- Scalar (0D): a number (or a single-entry dict is tolerated by the compiler)
- 1D over set I: {"i1": 1.0, "i2": 2.0, ...}
- 2D over sets I,J (RECOMMENDED): {"i1": {"j1": 1.0, "j2": 3.0}, "i2": {...}, ...}

3) vars
A list of decision variables. Each var is:
{
  "name": "string",
  "indices": [] | ["SetName"] | ["SetName1","SetName2"],  // 0D/1D/2D only
  "vartype": "continuous" | "integer" | "binary",
  "lb": number,
  "ub": number or null,
  "description": "string or null"
}

4) objective
{
  "name": "string",
  "sense": "min" or "max",
  "expr": "Python expression string",
  "description": "string or null"
}

5) constraints
A list of constraints. Each constraint is:
{
  "name": "string",
  "expr_lhs": "Python expression string",
  "sense": "<=" | ">=" | "==",
  "expr_rhs": "Python expression string",
  "description": "string or null"
}
"""



def build_generate_message(question_txt: str, ir: ModelIR) -> List[Dict[str, str]]:
    """
    生成将ModelIR转换为Gurobi模型的LLM message
    """
    ir_str = "null"
    if ir is not None:
        # 取ir中的sets/params/vars/obj/constrs部分上传
        ir_dict = {
            "sets": [s.__dict__ for s in ir.sets],
            "params": [p.__dict__ for p in ir.params],
            "vars": [v.__dict__ for v in ir.vars],
            "objective": ir.objective.__dict__ if ir.objective is not None else None,
            "constraints": [c.__dict__ for c in ir.constraints],
        }
        ir_str = json.dumps(ir_dict, ensure_ascii=False, indent=2)

    system_prompt = BASE_SYSTEM_PROMPT

    user_prompt = f"""{SCHEMA_FOR_IR}
    
    TASK EXECUTION:
    1. PROBLEM DESCRIPTION:
    {question_txt}

    2. PROVIDED INTERMEDIATE REPRESENTATION (IR):
    {ir_str}

    CRITICAL REQUIREMENTS:
    - Generate a Python function named `build_gurobi_model()` that takes no parameters and returns the Gurobi model
    - The function should use the provided ModelIR to build the model
    - Include all necessary imports
    - At the end of the code, call `build_gurobi_model()` to create the model
    - Do NOT include any extra explanations outside the code

    YOUR OUTPUT MUST BE A VALID python code that creates a Gurobi model.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return messages



def _clean_generated_code(code: str) -> str:
    """清理代码字符串"""
    if not code:
        return ""
    
    # 去除markdown代码块标记
    if '```python' in code:
        parts = code.split('```python')
        if len(parts) > 1:
            code = parts[1].split('```')[0]
    elif '```' in code:
        parts = code.split('```')
        if len(parts) > 1:
            code = parts[1].split('```')[0]
    
    return code.strip()


def llm_to_gurobi(model_generate_code: str) -> Optional[gp.Model]:
    """
    将LLM生成的代码安全地转换为Gurobi模型
    """
    if not model_generate_code or not isinstance(model_generate_code, str):
        return None
    
    try:
        # 清理代码
        cleaned_code = _clean_generated_code(model_generate_code)
        # print("清理后的代码:", cleaned_code)
        
        # 创建安全的执行环境
        safe_globals = _create_safe_environment()
        
        # 执行代码
        exec_locals = {}
        exec(cleaned_code, safe_globals, exec_locals)
        
        # 提取模型对象 - 优先使用固定函数名
        return _extract_model_with_fixed_function(exec_locals)
        
    except SyntaxError as e:
        warnings.warn(f"代码语法错误: {e}")
        return None
    except gp.GurobiError as e:
        warnings.warn(f"Gurobi错误: {e}")
        return None
    except Exception as e:
        warnings.warn(f"执行错误: {type(e).__name__}: {e}")
        return None


def _extract_model_with_fixed_function(exec_locals: dict) -> Optional[gp.Model]:
    """从执行环境中提取模型对象 - 使用固定函数名"""
    # 方式1：直接调用固定函数名
    fixed_function_names = ['build_gurobi_model', 'create_gurobi_model', 'create_model']
    
    for func_name in fixed_function_names:
        if func_name in exec_locals:
            func = exec_locals[func_name]
            if callable(func):
                try:
                    result = func()
                    if isinstance(result, gp.Model):
                        return result
                except Exception as e:
                    warnings.warn(f"调用函数 {func_name} 失败: {e}")
                    # 继续尝试其他方式
    
    # 方式2：查找直接创建的模型变量（向后兼容）
    for name in ['model', 'm', 'mdl', 'gurobi_model', 'opt_model']:
        if name in exec_locals:
            obj = exec_locals[name]
            if isinstance(obj, gp.Model):
                return obj
    
    # 方式3：查找所有Model实例
    for var_name, var_value in exec_locals.items():
        if isinstance(var_value, gp.Model):
            return var_value
    
    return None


def _create_safe_environment() -> dict:
    """创建安全的执行环境"""
    # 创建一个受限的 __builtins__，只包含安全的函数
    safe_builtins = {
        'range': range,
        'len': len,
        'sum': sum,
        'min': min,
        'max': max,
        'abs': abs,
        'int': int,
        'float': float,
        'bool': bool,
        'str': str,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'enumerate': enumerate,
        'zip': zip,
        'isinstance': isinstance,
        'print': print,
        '__import__': __import__,
    }
    
    # 创建安全环境
    safe_globals = {
        '__builtins__': safe_builtins,
    }
    
    # 预先导入gurobipy，这样生成的代码可以直接使用
    safe_globals['gp'] = gp
    safe_globals['GRB'] = getattr(gp, 'GRB', None)
    
    # 添加 quicksum 函数以便使用
    if hasattr(gp, 'quicksum'):
        safe_globals['quicksum'] = gp.quicksum
    
    return safe_globals