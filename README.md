# IR2Solve — Intermediate-Representation-first Autoformulation for Optimization

**IR2Solve** is a research-oriented framework that bridges **natural-language optimization problems** and **executable mathematical programming models** under a **strict LLM budget**. It implements an **IR-first** pipeline:
- The LLM outputs a **schema-constrained intermediate representation (IR)** as JSON.
- A deterministic **IR→Solver compiler** converts the IR into a **Gurobi** model through a controlled evaluation environment.
- A **layered verifier (L1/L2/L3)** performs failure-mode-driven, mostly deterministic repairs—improving success rate without requiring multi-round agent reflection.

---

## Contents

This repository contains a minimal reference implementation used to reproduce the paper’s pipeline and benchmark runs.

### Code layout

```text
IR2Solve/
├── ir2solve_ir.py                 # ModelIR schema + deterministic IR→Gurobi compiler
├── ir2solve_nl2ir.py              # NL → IR prompting + robust JSON parsing
├── ir2solve_pipeline.py           # End-to-end pipeline orchestration
├── ir2solve_verifier_core.py      # Verifier framework + issue/repair reporting
├── ir2solve_verifier_layer1.py    # L1: build-safety & index hygiene checks
├── ir2solve_verifier_layer2.py    # L2: generic semantic sanity checks
├── ir2solve_verifier_layer3.py    # L3: optional type-aware rescue + acceptance tests
├── run_nl2ir_demo.py              # Single-instance demo
├── run_nl4opt_benchmark.py        # NL4Opt benchmark runner (directory dataset)
└── run_complexlp_benchmark.py     # ComplexLP benchmark runner (jsonl dataset)
```

## Requirements

- **Python**: 3.9+ (recommended: 3.10)
- **OpenAI API key**: required for NL → IR generation
- **Gurobi** + valid license: required for compilation and solving

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows PowerShell

pip install -r requirements.txt
```


