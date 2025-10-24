# Contributing Guidelines

These are the collaboration rules for our master's thesis repo.
The goal: keep `main` clean, keep results reproducible, and make it easy for everyone to follow what changed.

---

## Branching model

We use three types of branches:

- `main`
  - Stable, "presentation-ready". This is what we would show to our supervisor or include in the thesis appendix.
  - Protected: do not push directly.

- `dev`
  - Integration branch.
  - New work lands here first, after review.

- Feature / topic branches
  - Your day-to-day work.
  - Naming convention:
    - `feature/<short-description>-<name>` for new functionality
    - `fix/<short-description>-<name>` for bug fixes
    - `doc/<short-description>-<name>` for thesis writing / documentation
    - `exp/<short-description>-<name>` for a specific experiment run

  Examples:
  - `feature/data-cleaning-andreas`
  - `exp/baseline-demand-forecast-jorgen`
  - `doc/literature-review-torger`

**Flow:**
1. Branch off `dev`.
2. Do work / commit.
3. Open a Pull Request (PR) back into `dev`.
4. Someone else reviews and approves.
5. Merge.

Periodically, when `dev` is stable, we merge `dev` → `main`.

---

## Commits

Please use clear commit messages so the history is readable.
Recommended format (inspired by Conventional Commits):

- `feat: add baseline linear regression model`
- `fix: handle missing timestamps in preprocess`
- `docs: update literature review section on demand forecasting`
- `refactor: move training loop to src/thesis_project/training.py`

Avoid commits like `final fix`, `wip`, `aaa`, etc.

Small commits are good. Commit often.

---

## Pull Requests (PRs)

Every PR should briefly include:
1. **What changed**  
   ("Added preprocessing script that cleans missing hours in production data and writes parquet to data/processed/")
2. **How to reproduce / test**  
   (e.g. "Run `python scripts/preprocess_data.py --input data/raw/prod_oct.csv`")
3. **Open questions / assumptions**  
   ("We currently forward-fill missing demand. OK with that?")

Rules:
- PRs go into `dev`, not directly to `main`.
- No self-merge unless at least one other person has approved or commented "LGTM ✅".
- If you pair-programmed with someone, that counts as review.

---

## Experiments

When you run an experiment, please:
- Create a folder `experiments/expXYZ_description/`
- Include:
  - `config.yaml` (hyperparameters, data slice, etc.)
  - `results.csv` or `metrics.json`
  - `notes.md` (short bullet summary: goal, result, key observations)

This is how we'll later write the Results and Discussion chapters without guessing.

---

## Data policy

- Do **not** commit real/raw data into Git.
- Do **not** commit any credentials / API keys / access tokens.
- If you create synthetic toy data for testing, that's fine to commit as long as it’s <1MB and non-sensitive.
- Document data sources and preprocessing steps in `data/README.md`.

---

## Coding style

- Put reusable logic into `src/thesis_project/`, not only inside notebooks.
- Notebooks in `notebooks/` are allowed to be messy/interactive, but:
  - The notebook filename should start with `YYYY-MM-DD-name-topic.ipynb` so we understand timeline and owner.
  - If something in a notebook becomes "the real method", move it into `src/` and import it instead of copy/pasting cells everywhere.

---

## Documentation

Where to write things:
- Meeting notes → `docs/meeting_notes/DATE_topic.md`
- Project plan / timeline / next steps → `docs/project_plan.md`
- Thesis draft text / figures → `reports/`

Assumptions and reasoning are just as valuable as code. Write them down.

---

## Tests

If you add new core logic in `src/thesis_project/...`, please also add/update tests in `tests/`.

The goal isn’t perfect coverage. The goal is:
- We don't silently break preprocessing / training right before a deadline.
- We can rerun old experiments and trust the pipeline still works.

---

## TL;DR

1. Work on a feature branch off `dev`.
2. Commit with useful messages.
3. Open a PR back into `dev`.
4. Someone else reviews.
5. Record experiments.
6. No data / secrets in Git.

Thanks :)
