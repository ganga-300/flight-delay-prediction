# Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning

> **Project 21 — Phase 1 (Machine Learning).** Predict whether a flight will **arrive ≥ 15 minutes late**
> using only information available **before departure**, with calibrated probabilities.

## Status

An end-to-end flight delay prediction project with trained Logistic Regression and XGBoost models, probability calibration, evaluation reports, and automated tests.

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

- **Target:** `is_delayed = 1` if arrival delay ≥ 15 min (BTS on-time convention), else 0. Cancelled flights are excluded from the modelling dataset.
- **Decision time:** D-1 (the day before departure), as configured in `configs/config.yaml`. Every feature must be computable at that moment — see `docs/leakage_policy.md`.
- **Split:** chronological by date (train → validation → test). No random shuffling.
- **Metrics:** PR AUC (primary, class-imbalanced), ROC AUC, recall at a chosen operational precision, Brier score, ECE and reliability diagrams.

## Dataset

Kaggle *Flights* dataset (2013 US flights departing NYC). Larger extension: BTS Airline On-Time Performance.
Raw data is **not** committed. See `data/README.md` for download instructions.

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

## Current workflow

Activate the virtual environment and run the test suite:

```bash
source .venv/bin/activate
pytest -q
```

### Evaluation results

Held-out test-set results:

| Model | PR-AUC | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.3962 | 0.6619 |
| XGBoost | 0.3887 | 0.6540 |

Logistic Regression performs slightly better on both reported metrics. PR-AUC is useful for evaluating performance when the classes are imbalanced.

Detailed reports are available in `reports/results/`, including test metrics, threshold analysis, calibration comparisons, feature ablation, and monthly test metrics.

### Important limitation

The delay rate differs between the validation and test periods. This temporal distribution shift can affect threshold selection and generalisation. Thresholds selected on validation data should be evaluated carefully before operational use.

## Phase 2 (later)

Compact MLP with learned embeddings for airline / origin / destination, optional temporal model,
and an offline what-if rescheduling analysis. Not part of this scaffold.

## Author

Ganga Raghuwanshi — Newton School of Technology, Rishihood University
