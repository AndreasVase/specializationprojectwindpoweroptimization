# Multimarket, multistage stochastic optimization model for optimal bidding strategy of wind power in Norway (Working Title)

## Project Summary
This repository contains the code, experiments, documentation, and writing for our master's thesis project (NTNU / Industrial Economics and Technology Management, 2025).

**Research question (draft):**  
We develop a multimarket, multistage stochastic optimization model for optimal bidding of wind power in the Norwegian electricity market.  
The model incorporates market uncertainty, production variability, and system constraints to evaluate different bidding strategies under varying scenarios.

We will update this section as the project evolves.

---

## Team
- **Andreas Buer Vase**  
- **Jørgen Stenersen**  
- **Torger Skrettingland**

**Supervisors:**  
- Carl Henrik Andersson  
- Odd Erik Gundersen (co-supervisor)

---

## Repository Structure

```
.
├─ README.md               # You are here
├─ CONTRIBUTING.md         # Collaboration rules
├─ .gitignore              # Files/dirs we don't commit
├─ requirements.txt        # Python dependencies
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
├─ scripts/                # CLI-style scripts to run preprocessing, optimization, etc.
│  ├─ preprocess_data.py
│  ├─ run_optimization.py
│  └─ evaluate_results.py
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
```

---

## How to set up your local environment

We use a standard Python virtual environment and a shared `requirements.txt` file.

### 1. Clone the repository
```bash
git clone <repo-url>
cd master-thesis
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
```

### 3. Activate the environment
```bash
# On macOS / Linux:
source .venv/bin/activate

# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

### 4. Upgrade pip (recommended)
```bash
pip install --upgrade pip
```

### 5. Install dependencies
```bash
pip install -r requirements.txt
```

### 6. (Optional) Add Jupyter kernel
To use this environment inside JupyterLab:
```bash
python -m ipykernel install --user --name thesis --display-name "Thesis"
```

---

## Data access

Raw and processed data are **not** committed to Git.  
Each collaborator must store data locally in the `/data/raw` and `/data/processed` folders.  
See `data/README.md` for dataset descriptions and access instructions.

---

## How to run experiments

Example workflow:
```bash
# Preprocess data
python scripts/preprocess_data.py

# Run optimization model
python scripts/run_optimization.py --config experiments/exp001_baseline/config.yaml

# Evaluate results
python scripts/evaluate_results.py --input experiments/exp001_baseline/results.csv
```

Each experiment should have its own folder in `/experiments/` with:
- a `config.yaml` (input parameters)
- `results.csv` or `metrics.json` (outputs)
- `notes.md` (purpose, findings, and comments)

---

## Branching Workflow

We use the following structure for collaboration:

- `main` → stable, presentation-ready code  
- `dev` → integration branch where new work is merged  
- `feature/<short-desc>-<name>` → individual branches for tasks  

**Example branch names:**
```
feature/model-formulation-andreas
exp/scenario-generation-jorgen
doc/lit-review-torger
```

Work off `dev`, open Pull Requests, and merge after peer review (see `CONTRIBUTING.md` for details).

---

## Reproducibility Philosophy

- Every result should be reproducible from code + configuration.  
- Each experiment must be documented.  
- No data or secrets should ever be pushed to Git.  
- All assumptions, constraints, and observations should be recorded in `docs/` or the experiment folder.

---

## License
This project is intended for academic use within NTNU.  
Do not redistribute data or code without permission from the authors and supervisors.


All assumptions, constraints, and observations should be recorded in docs/ or the experiment folder.
