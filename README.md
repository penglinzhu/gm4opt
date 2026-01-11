# IR2Solve — Intermediate-Representation-first Autoformulation for Optimization

**IR2Solve** is a research-oriented framework that bridges **natural-language optimization problems** and **executable mathematical programming models** under a **strict LLM budget**. It implements an **IR-first** pipeline:
- The LLM outputs a **schema-constrained intermediate representation (IR)** as JSON.
- A deterministic **IR→Solver compiler** converts the IR into a **Gurobi** model through a controlled evaluation environment.
- A **layered verifier (L1/L2/L3)** performs failure-mode-driven, mostly deterministic repairs—improving success rate without requiring multi-round agent reflection.
---

## End-to-End Pipeline

IR2Solve implements a unified pipeline:

**NL → (IR-first system prompting) → IR (JSON) → ModelIR → (Layered Verifier L1/L2/L3) → IR→Gurobi Compilation → Solve & Log**

- **NL**: raw natural-language optimization problem statement.
- **IR-first system prompting**: a schema-driven prompting strategy that constrains the LLM to emit a compile-ready IR (JSON) rather than free-form code/LaTeX.
- **IR (JSON)**: a strictly structured intermediate representation emitted by the LLM, consisting of `sets/params/vars/objective/constraints`.
- **ModelIR**: a typed, schema-aligned in-memory dataclass representation parsed from the JSON IR, enabling static checks and deterministic transformations.
- **Layered Verifier (L1/L2/L3)**: failure-mode-driven repair operators applied to the IR/ModelIR:
  - **L1 (Compile-Safety & Index Hygiene)**: deterministic normalization/repairs to prevent build-time errors (e.g., canonicalizing keys, fixing missing diagonals for square 2D parameters, unrolling free-index constraints into scalar constraints).
  - **L2 (Light Semantics)**: conservative generic semantic sanity checks (e.g., integrality sanity, constraint-direction sanity, token/normalization fixes).
  - **L3 (Conditional Structure Rescue, optional)**: type-guided IR rebuild triggered only under high confidence, guarded by acceptance tests (the rebuilt IR must pass compilation and basic solve checks to be adopted).
- **IR→Gurobi Compilation**: deterministic compilation of ModelIR into a Gurobi model; objective/constraints are compiled from executable expression strings under a restricted evaluation environment.
- **Solve & Log**: optimize the compiled model and write structured logs, including `failure_stage`, solver status/objective, and verifier `issues/repairs` traces for error analysis.

---

## Core Contributions

IR2Solve’s contributions target robust autoformulation under a strict LLM budget:

### A) IR-first Autoformulation with Executable Semantics

IR2Solve constrains the LLM output into a compile-ready intermediate representation (IR):

- The LLM emits **ModelIR JSON** with explicit `sets/params/vars/objective/constraints`.
- Objective/constraints are represented as **executable expression strings** and compiled by a deterministic **IR→Gurobi** compiler in a controlled environment.
- This reduces hallucinated APIs and glue-code errors common in code-first pipelines.

### B) Layered Verifier as Deterministic Repair Operators

IR2Solve uses a layered verifier that applies repairs only when needed:

- **L1** fixes compile-safety and index hygiene issues (build-time failures).
- **L2** performs conservative generic semantic sanity checks.
- **L3** optionally triggers a type-guided rebuild, but only under high confidence and only accepted after passing compilation/solve acceptance tests.

This design emphasizes **rule-based, measurable repairs** rather than unconditional multi-round reflection.

### C) Accuracy-per-Cost Optimization Under a Strict LLM Budget

IR2Solve is optimized for settings where LLM calls are expensive:

- Most improvements come from **deterministic verification and repairs** rather than extra LLM sampling.
- Optional **L3** uses a **high-confidence trigger + acceptance gate**, enabling rare second-call recovery without turning the pipeline into a high-cost exploration method.

---

## Project Structure

```text
IR2Solve/
├── ir2solve_ir.py                 # ModelIR schema + deterministic IR→Gurobi compiler (executable semantics)
├── ir2solve_nl2ir.py              # NL → IR: schema-driven prompting + robust JSON extraction/parsing
├── ir2solve_pipeline.py           # Pipeline orchestration (LLM → IR → verifier → compile → solve → logs)
├── ir2solve_verifier_core.py      # Verifier framework + orchestration + reporting
├── ir2solve_verifier_layer1.py    # L1 rules: compile-safety & index hygiene
├── ir2solve_verifier_layer2.py    # L2 rules: light semantic sanity checks
├── ir2solve_verifier_layer3.py    # L3 rules: optional type-guided rescue + acceptance testing
├── run_industryor_benchmark.py    # Benchmark runner (kept name) — evaluation + CSV/trace logging
├── run_complexlp_benchmark.py     # Benchmark runner (kept name) — evaluation + CSV/trace logging
└── README.md
```
---

## Requirements

- Python 3.9+ (recommended)
- OpenAI API key (for NL → IR)
- Gurobi + valid license (for solving)

Install dependencies:
```bash
pip install -r requirements.txt
