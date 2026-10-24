# Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning

> **Project 21 — Phase 1 (Machine Learning).** Predict whether a flight will **arrive ≥ 15 minutes late**
> using only information available **before departure**, with calibrated probabilities.

## Status

Data preparation, EDA and feature engineering are complete (59 leakage-safe features). Model,
metric and calibration modules are not implemented yet. The Nov–Dec test holdout is sealed during
development; every decision is recorded in `docs/decision_log.md`.

| Phase 1 criterion | Where it lives |
|---|---|
| Literature Review | `docs/literature_review.md`, `report/references.bib` |
| Dataset Quality & EDA | `notebooks/01_data_quality.ipynb`, `notebooks/02_eda.ipynb`, `docs/data_dictionary.md` |
| Feature Engineering | `src/flight_delay/features/`, `docs/leakage_policy.md` |
| Theoretical Rigor | `docs/methodology.md`, `report/sections/` |
| Model Application | `src/flight_delay/models/` (Logistic Regression, XGBoost, calibration) |
| Evaluation | `src/flight_delay/evaluation/` (PR AUC, ROC AUC, recall@precision, calibration) |
| GitHub Repo & Code Quality | `tests/`, `pyproject.toml` (ruff + pytest config) |
| Project Presentation | `presentation/` |
| Project Report (IEEE) | `report/` (IEEEtran LaTeX) |

## Problem definition

- **Target:** `is_delayed = 1` if arrival delay ≥ 15 min (BTS on-time convention), else 0. Cancelled flights and flights without an arrival record are excluded (D-010, D-011).
- **Decision time:** end of day D−1 (`decision_time.mode: d_minus_1`). Every feature must be computable at that moment — see `docs/leakage_policy.md` (D-012).
- **Split:** chronological by date — train Jan–Aug, validation Sep–Oct, test Nov–Dec 2013 (D-013). No random shuffling. The test period is sealed: development code reads a Jan–Oct file only (D-002).
- **Metrics:** PR AUC (primary, class-imbalanced), ROC AUC, recall at a chosen operational precision, Brier score, ECE and reliability diagrams.

## Dataset

Kaggle *Flights* dataset (2013 US flights departing NYC). Larger extension: BTS Airline On-Time Performance.
The raw file `data/raw/flights.csv` is committed; see `data/README.md` for provenance, hashes and the
sealing step that creates the Jan–Oct development file.

## Repository layout

```
configs/            experiment configuration (paths, decision time, split dates, target threshold)
data/               raw / interim / processed data (git-ignored contents)
docs/               literature review, data dictionary, leakage policy, methodology
notebooks/          numbered exploratory notebooks (EDA, feature checks, model results)
src/flight_delay/   importable package: data, features, models, evaluation, utils
scripts/            CLI entry points that chain the pipeline
tests/              pytest suite (leakage, split, features, metrics)
models/             trained model artifacts (git-ignored)
reports/            generated figures and metric tables
report/             IEEE-format LaTeX project report
presentation/       slides and speaker notes for the Phase 1 evaluation
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Pipeline (planned)

```bash
python scripts/seal_holdout.py       # Jan–Oct development file; Nov–Dec stays sealed
python scripts/build_features.py     # 59 features -> data/processed/{train,valid}.parquet
python scripts/train.py --model logreg   # TODO
python scripts/train.py --model xgboost  # TODO
python scripts/evaluate.py           # TODO
```

## Rules for contributors (training stage)

Work on a branch off `ml-restart` and open a pull request into `ml-restart`. Tests run
automatically and must pass.

- **Read data only through** `flight_delay.data.dev_splits`: `load_dev_split("train")`,
  `load_dev_split("valid")` and `feature_columns()`. The test split cannot be loaded.
- **Write the experiment plan first.** Fill in `configs/experiment_plan.yaml` (models and the
  exact hyperparameter values to try), set `status: approved`, and record it as D-201 in
  `docs/decision_log.md`. Until the owner approves that pull request, runs are refused.
- **Record every run** with `flight_delay.experiments.write_run_record`, on committed code, and
  commit the record from `experiments/runs/`. Only `train_` and `valid_` metrics are accepted.
- **Do not change** features, split dates, feature settings or the guards. They are frozen
  (tag `features-frozen`) and owner-reviewed (`.github/CODEOWNERS`).
- **The test set is evaluated once,** by `scripts/evaluate.py` through
  `flight_delay.data.holdout.open_sealed_holdout`, after the chosen model is tagged
  `model-frozen`. It cannot be re-run once results are committed.

## Phase 2 (later)

Compact MLP with learned embeddings for airline / origin / destination, optional temporal model,
and an offline what-if rescheduling analysis. Not part of this scaffold.

## Author

Ganga Raghuwanshi — Newton School of Technology, Rishihood University
