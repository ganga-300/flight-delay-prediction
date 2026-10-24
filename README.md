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

## Reporting-buffer sensitivity analysis

Historical features were evaluated with reporting buffers of 0, 30, 60 and 120 minutes. These are sensitivity experiments, not proof that any buffer reflects actual data availability.

### Logistic Regression

| Buffer | Test PR-AUC | Test ROC-AUC | Brier | ECE | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0 min | 0.39166 | 0.66001 | 0.18805 | 0.08787 | 0.5501 | 0.04331 |
| 30 min | 0.38992 | 0.65873 | 0.18852 | 0.08936 | 0.5291 | 0.05349 |
| 60 min | 0.38870 | 0.65739 | 0.18917 | 0.09166 | 0.5395 | 0.04998 |
| 120 min | 0.38674 | 0.65499 | 0.18995 | 0.09385 | 0.5183 | 0.05887 |

### XGBoost

| Buffer | Test PR-AUC | Test ROC-AUC | Brier | ECE | Validation threshold meets 60% precision? |
|---:|---:|---:|---:|---:|---|
| 0 min | 0.38381 | 0.64804 | 0.19102 | 0.09619 | Yes |
| 30 min | 0.38518 | 0.64999 | 0.19067 | 0.09591 | No |
| 60 min | 0.38206 | 0.64682 | 0.19105 | 0.09590 | No |
| 120 min | 0.38128 | 0.64542 | 0.19170 | 0.09801 | No |

The 30-minute XGBoost run achieved the highest test PR-AUC and ROC-AUC among these configurations, but the differences were small. Only the 0-minute run found a validation threshold meeting the 60% precision target. Since the test set was evaluated across these configurations, these comparisons are exploratory and should not be used for further test-set tuning.

The reporting buffer is an assumption about historical outcome availability. Its operational validity depends on the actual data source and reporting latency, which have not been independently verified.
