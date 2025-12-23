# GM4OPT — NL→IR→Solver Pipeline for Optimization Modeling

GM4OPT（graph modeling for optimization） is a research-oriented framework that bridges **natural-language optimization problems** and **executable mathematical programming models**. It provides a unified pipeline that (i) translates problem text into a structured intermediate representation (IR) and (ii) deterministically compiles the IR into a solver-ready model (Gurobi), with optional graph-based analysis modules.

---

## End-to-End Pipeline

GM4OPT implements a unified pipeline:

**NL → (dynamic system prompt templating) → IR (JSON) → ModelIR → (GraphIR → Difficulty) → Gurobi**

- **NL**: raw natural-language problem statement.
- **Dynamic system prompt templating**: a heuristic prompt-construction mechanism that augments a fixed base prompt with problem-type-specific guidance.
- **IR (JSON)**: a strictly structured intermediate representation emitted by the LLM.
- **ModelIR**: typed, validated in-memory representation parsed from the JSON IR (schema-driven).
- **GraphIR (optional)**: a graph representation derived from IR/ModelIR to support structural checks and analysis.
- **Difficulty (optional)**: heuristic difficulty estimation derived from structural signals (e.g., number of sets/variables/constraints, graph statistics).
- **Gurobi**: deterministic model compilation and optimization.

---

## Core Contributions

### 1) Structured IR and a Deterministic NL→IR→Solver Bridge
GM4OPT designs a **structured, schema-constrained IR** for linear / integer optimization problems and provides a complete bridge:
- **NL → IR**: LLM produces a JSON object following a strict schema.
- **IR → Solver**: GM4OPT compiles the IR into a solver-executable optimization model through a fixed, deterministic procedure.

This decomposition separates the inherently uncertain part (NL → model extraction) from the deterministic part (model execution), improving debuggability and enabling systematic evaluation.

### 2) Dynamic System Prompt Templating for NL→IR
Instead of using a fully static prompt, GM4OPT adopts **dynamic system prompt templating**:
- A **fixed base prompt** enforces schema rules and modeling constraints (e.g., fully unrolled scalar constraints, no implicit quantifiers).
- A **dynamic guidance block** is generated heuristically by extracting keywords from the problem text and classifying the problem type (e.g., assignment, transport/flow, routing, production mix).
- The final system prompt is **base + type-specific heuristics**, aiming to reduce common NL→IR errors (e.g., wrong variable types, wrong constraint senses, missing structural constraints).

This component is intentionally lightweight and heuristic: it does not rely on retrieval or learned prompt selection, but on rule-based feature extraction and problem-type templates.

### 3) Graph-Based IR Instrumentation for Self-Check and Difficulty Estimation (Experimental)
GM4OPT enriches the IR processing stage with a **graph-structured representation**:
- The pipeline can construct a **GraphIR** that connects entities such as sets, parameters, variables, constraints, and objectives.
- This graph supports:
  - **Structural self-checks** (e.g., connectivity, orphan variables, unused parameters, index consistency signals).
  - **Difficulty estimation** based on structural and graph statistics.

Both **graph-based self-check** and **difficulty estimation** are currently **experimental modules**. A key ongoing direction is to integrate them more tightly into the overall architecture—specifically, to make graph/difficulty feedback *actionable*, so that it can guide and refine IR generation (e.g., prompting revisions, identifying missing constraint families, correcting modeling patterns).

---

## Project Structure

```text
gm4opt/
├── gm4opt_ir.py           # ModelIR schema definitions (Meta/Sets/Params/Vars/Objective/Constraints)
├── gm4opt_nl2ir.py        # NL → IR: prompt construction (dynamic templating) + JSON extraction/parsing
├── gm4opt_pipeline.py     # Unified pipeline orchestration (LLM → IR → ModelIR → optional Graph/Difficulty → Gurobi)
├── gm4opt_graph.py        # GraphIR construction + structural checks (experimental)
├── gm4opt_difficulty.py   # Difficulty scoring based on IR/Graph features (experimental)
├── run_nl2ir_demo.py      # Minimal demo: run pipeline on a single NL instance
├── run_*_benchmark.py     # Benchmark runners: batch evaluation + CSV logging + IR dumps
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
