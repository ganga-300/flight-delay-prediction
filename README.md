# Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning

> **Project 21 — Phase 1 (Machine Learning).** Predict whether a flight will **arrive ≥ 15 minutes late**
> using only information available **before departure**, with calibrated probabilities.

## Status

Phase 1 is complete. Data preparation, EDA, feature engineering (59 leakage-safe features),
model training and selection, probability calibration and a single evaluation on the sealed
November–December test set are done. Every decision, with its evidence and date, is recorded in
`docs/decision_log.md` (D-000 to D-205).

## Final results (test set, November–December 2013, evaluated once)

Model: L2 Logistic Regression (`C = 0.01`, `class_weight = "balanced"`), selected on August,
retrained on January–August, sigmoid-calibrated on September (D-202, D-204, D-205).

| Probabilities | Flights | Delay rate | PR-AUC | ROC-AUC | Brier | ECE (10 bins) |
|---|---|---|---|---|---|---|
| Raw | 53,991 | 0.258 | 0.398 | 0.664 | 0.226 | 0.204 |
| Calibrated (final) | 53,991 | 0.258 | 0.398 | 0.664 | 0.185 | 0.077 |

- The model ranks delayed flights about 1.5 times better than random (PR-AUC 0.398 against a
  0.258 delay rate), consistent with validation.
- Calibration cuts ECE from 0.204 to 0.077. Calibrated probabilities run low on the test period
  (0.18 predicted vs 0.26 observed) because calibration was fitted on a calmer September.
- No threshold reached the 60% precision target on October (D-204), so threshold-based metrics
  and the rescheduling rule are reported as not available. The target was not lowered.
- Limitations: no weather or same-day information at decision time; validation (Sep–Oct) is
  calmer than the test period. An earlier iteration of this project evaluated on the test set;
  every result here was re-developed without reference to it (D-000).

| Phase 1 criterion | Where it lives |
|---|---|
| Literature Review | `docs/literature_review.md`, `report/references.bib` |
| Dataset Quality & EDA | `notebooks/01_data_quality.ipynb`, `notebooks/02_eda.ipynb`, `docs/data_dictionary.md` |
| Feature Engineering | `src/flight_delay/features/`, `docs/leakage_policy.md` |
| Theoretical Rigor | `docs/methodology.md`, `report/sections/` |
| Model Application | `src/flight_delay/models/`, `scripts/train.py`, `models/d201/` (D-201, D-202) |
| Evaluation | `src/flight_delay/evaluation/`, `scripts/calibrate_threshold.py`, `scripts/evaluate.py`, `reports/`, `notebooks/04_model_results.ipynb` (D-203 to D-205) |
| GitHub Repo & Code Quality | `tests/`, `pyproject.toml` (ruff + pytest config) |
| Project Presentation | `presentation/` |
| Project Report (IEEE) | `report/FlightDelayPrediction.pdf` (compiled), `report/` (IEEEtran LaTeX source) |

## Problem definition

- **Target:** `is_delayed = 1` if arrival delay ≥ 15 min (BTS on-time convention), else 0. Cancelled flights and flights without an arrival record are excluded (D-010, D-011).
- **Decision time:** end of day D−1 (`decision_time.mode: d_minus_1`). Every feature must be computable at that moment — see `docs/leakage_policy.md` (D-012).
- **Split:** chronological by date — train Jan–Aug, validation Sep–Oct, test Nov–Dec 2013 (D-013). No random shuffling. The test period is sealed: development code reads a Jan–Oct file only (D-002).
- **Metrics:** PR AUC (primary, class-imbalanced), ROC AUC, recall at a chosen operational precision, Brier score, ECE and reliability diagrams.

## Dataset

Kaggle [*Flights*](https://www.kaggle.com/datasets/matinsajadi/flights) dataset (2013 US flights
departing NYC), provided by the course. Larger extension: BTS Airline On-Time Performance.
The raw file `data/raw/flights.csv` is committed; see `data/README.md` for provenance, hashes and the
sealing step that creates the Jan–Oct development file.

## Repository layout

```
configs/            experiment configuration (paths, decision time, split dates, target threshold)
data/               raw/flights.csv (committed); interim/ and processed/ are generated, not committed
docs/               literature review, data dictionary, leakage policy, methodology
notebooks/          01 data quality, 02 EDA (Jan–Aug), 03 feature checks, 04 results summary (read-only)
src/flight_delay/   importable package: data, features, models, evaluation, utils
scripts/            CLI entry points that chain the pipeline
tests/              pytest suite (leakage, split, sealed test set, features, metrics, calibration, training CLI)
models/             trained models: d201/ (final model, run logs), d203/ (calibration, operating point)
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

## Pipeline

```bash
python scripts/seal_holdout.py              # Jan–Oct development file; Nov–Dec stays sealed
python scripts/build_features.py            # 59 features -> data/processed/{train,valid}.parquet
python scripts/train.py --model all --execute   # D-201: fit Jan–Jul, tune Aug, retrain Jan–Aug
python scripts/report_d201.py               # D-202: August comparison and feature importance
python scripts/calibrate_threshold.py       # D-204: calibrate on Sep, threshold on Oct
python scripts/evaluate.py                  # D-205: single test evaluation
```

The first two steps rebuild the data on any machine. The last four have already been run, and
their outputs are committed (`models/d201/`, `models/d203/`, `reports/`). `train.py` and
`calibrate_threshold.py` refuse to overwrite their outputs. `evaluate.py` is single-use: it refuses
to run once `reports/results/test_metrics.json` exists.

## Phase 2 (later)

Compact MLP with learned embeddings for airline / origin / destination, optional temporal model,
and an offline what-if rescheduling analysis. Not part of Phase 1.

## Author

Ganga Raghuwanshi — Newton School of Technology, Rishihood University
