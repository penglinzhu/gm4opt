# IR2Solve — Intermediate-Representation-first Autoformulation for Optimization with Cost-Efficient LLM Usage

**IR2Solve** is a research-oriented framework that bridges **natural-language optimization problems** and **executable mathematical programming models** with **cost-efficient LLM usage**. It implements an **IR-first** pipeline:
- The LLM outputs a **schema-constrained intermediate representation (IR)** as JSON.
- A **layered verifier (L1/L2/L3)** performs failure-mode-driven, mostly deterministic repairs—improving success rate without requiring multi-round agent reflection.
- A deterministic **IR→Solver compiler** converts the IR into a **Gurobi** model through a controlled evaluation environment.

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
pip install -r requirements.txt
```

### Configure OpenAI API key
Linux/Mac:
```bash
export OPENAI_API_KEY="YOUR_KEY"
```
Windows PowerShell:
```powershell
setx OPENAI_API_KEY "YOUR_KEY"
```

### Configure Gurobi
Make sure your Gurobi installation and license are available

## Quickstart
### Run a single-instance demo
```bash
python run_nl2ir_demo.py
```
This runs a built-in example and prints the pipeline outcome.
To test your own problem statement, edit the QUESTION_TEXT string in run_nl2ir_demo.py.

### Run NL4Opt (directory dataset)
```bash
python run_nl4opt_benchmark.py
```

### Run ComplexLP (jsonl dataset)
```bash
python run_complexlp_benchmark.py
```

### Outputs and evaluation

Across benchmarks, the main outputs are:

- **`*_results.csv`**: per-instance results (one row per problem)
- **`*_trace.jsonl`**: structured per-instance trace (one JSON object per line)
- **`*_summary.txt`**: aggregate statistics (accuracy, token/call usage, failure breakdown)
- **`ir_outputs_*/`**: saved final IR JSONs (one file per instance)

## Data

All datasets required for reproduction are **included in this repository** under `./data/` (no external download needed).

- `run_nl4opt_benchmark.py` reads NL4Opt instances from `data/NL4Opt/`
- `run_complexlp_benchmark.py` reads ComplexLP from `data/Mamo/Mamo_complex_lp_clean.jsonl`

## Citation
Please cite the accompanying paper (submitted under double-blind review) if you use this code.
