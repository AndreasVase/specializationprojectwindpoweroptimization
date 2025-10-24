# Multimarket, multistage stochastic optimization model for optimal bidding strategy of wind power in Norway (Working Title)

## Project Summary
This repository contains the code, experiments, documentation, and writing for our master's thesis project (NTNU / [Your Program Here], 2025).

**Research question (draft):**  
Describe in a few sentences what we are trying to do.  
Example: "We build and evaluate optimization / forecasting models to solve [problem]. We compare multiple approaches, document assumptions, and evaluate performance using real-world data."

We will update this once the problem statement is finalized.

---

## Team
- Andreas Buer Vase  
- Jørgen Stenersen
- Torger Skrettingland

Supervisors:
- Carl Henrik Andersson
- Odd Erik Gundersen (co-supervisor)

---

## Repository Structure
```text
.
├─ README.md               # You are here
├─ CONTRIBUTING.md         # Collaboration rules
├─ .gitignore              # Files/dirs we don't commit
├─ environment.yml         # Reproducible Python environment (conda), WIP
│
├─ data/
│  ├─ raw/                 # Original data dumps (not committed)
│  ├─ processed/           # Cleaned / feature-ready data (not committed)
│  └─ README.md            # Where data comes from and how to access it
│
├─ src/
│  └─ thesis_project/      # Python source code (modules, utils, pipelines)
│     └─ __init__.py
│
├─ scripts/                # CLI-style scripts to run preprocessing, training, etc.
│  ├─ preprocess_data.py
│  ├─ train_model.py
│  └─ evaluate.py
│
├─ notebooks/              # Jupyter notebooks for exploration / EDA / prototyping
│  ├─ 2025-10-24-andreas-baseline-model.ipynb
│  └─ ...
│
├─ experiments/            # Each experiment gets its own folder with config + results
│  ├─ exp001_baseline/
│  │   ├─ config.yaml
│  │   ├─ results.csv
│  │   └─ notes.md
│  └─ ...
│
├─ reports/
│  ├─ figures/             # Plots, generated figures for thesis
│  └─ paper/               # Draft thesis text (LaTeX / Overleaf exports / etc.)
│
├─ docs/
│  ├─ project_plan.md      # Scope, timeline, milestones
│  └─ meeting_notes/       # Supervisor and group meeting notes
│     ├─ 2025-10-24_kickoff.md
│     └─ ...
│
└─ tests/                  # Unit tests for src/
   └─ test_example.py


