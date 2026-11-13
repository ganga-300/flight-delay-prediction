# Decision Log

Every modelling decision is recorded here **before** it is acted on. Each entry states what was
decided, the options considered, the evidence and which data it came from, and whether it was made
before or after any result was seen. A choice that is not in this log should not be in the
project.

Data periods: **train** Jan–Aug 2013, **validation** Sep–Oct 2013, **test** Nov–Dec 2013 (sealed).

Status values: **Fixed** (decided, not to be revisited on results), **Open** (must be decided
before the phase that needs it), **Assumption** (cannot be verified from this dataset; reported as
a limitation).

---

## Phase 0 — Restart and test-set protection

### D-000 · Disclosure of an earlier iteration
- **Fact:** An earlier iteration of this project (branch `main`, tag `alisha-work-snapshot`,
  commit `d4d5da3`) evaluated models on the Nov–Dec test period several times. That iteration
  also changed XGBoost hyperparameters and removed feature groups after viewing test results.
- **Decision:** None of that iteration's model code, hyperparameters, feature-group choices or
  results is reused. The modelling stage is redeveloped from scratch on branch `ml-restart`
  (renamed `main` on 2026-10-25; the earlier iteration is kept as `old-main`).
  Every model, feature and threshold reported is selected without reference to those results.
- **Report wording:** "An earlier iteration of this project evaluated on the Nov–Dec holdout; all
  models and features reported here were re-developed and selected without reference to those
  results."
- **Status:** Fixed. Date: 2026-10-25.

### D-001 · Restart point
- **Decision:** Restart from commit `648b780`, the last commit before the earlier iteration's
  modelling work. It contains data loading, cleaning, the target, the chronological split and the
  feature pipeline. Model, metric and calibration modules are unimplemented stubs.
- **Evidence:** The feature pipeline at `648b780` passes its tests (32 passed, 2 skipped as
  unimplemented). A rebuild with every Nov–Dec row removed from the raw data produces identical
  features: 0 differing cells across all 273,355 Jan–Oct rows × 59 features + target, and an
  identical fitted encoder. So no Nov–Dec value reaches any train or validation feature.
- **Known caveat:** Some feature *design* choices were informed by an EDA over all 12 months
  (`notebooks/02_eda.ipynb`). These are re-checked on train data in Phase 2 (see D-101).
- **Status:** Fixed. Date: 2026-10-25.

### D-002 · Sealed test holdout
- **Decision:** All development (EDA, features, training, selection) uses a Jan–Oct file only.
  The Nov–Dec rows are read **once**, by the final test evaluation after the model is frozen.
- **Mechanism:**
  - `scripts/seal_holdout.py` copies every Jan–Oct line of the full file byte-for-byte into
    `data/interim/flights_dev.csv`. It reads only each line's `month` field; no Nov–Dec delay,
    time or label value is parsed.
  - `configs/config.yaml`: `paths.raw` is the dev file and `paths.raw_sealed` the full file.
  - `data.load.load_dev_raw` raises if any row is dated after 2013-10-31.
    `scripts/build_features.py` and the real-data tests load through it.
  - `tests/test_holdout_seal.py` fails if development code references the sealed file or the test
    split, or if the dev data holds a holdout row.
  - The build writes no test split file. The full-year file left by an earlier build was removed.
- **Hashes (SHA-256):**
  - Sealed full file `data/raw/flights.csv` (336,776 rows):
    `8bdc6abbf07ac0cbb1036bc14a787cab3eea616be46c31899d2d8048963b08aa`
  - Dev file `data/interim/flights_dev.csv` (281,373 rows, Jan–Oct):
    `8f206680eda4dc42098dbd4e9a6ec6f89f468afc41cbb913fd528fa4fc0018df`
  - Rows withheld (Nov–Dec): 55,403
- **Check after sealing:** train 217,727 rows (26.37% delayed) and validation 55,628 rows (15.70%),
  identical to the full-year build.
- **Why history features still work at final evaluation:** Rolling history is computed per row from
  outcomes known by end of D−1, so the final run over the full year gives test rows the same
  history a live system would have. D-001 shows this does not change any train/validation value.
- **Status:** Fixed. Date: 2026-10-25.

### D-003 · Random seed
- **Decision:** `seed: 42` in `configs/config.yaml`, used for every stochastic step.
- **Reason:** Reproducibility. The value is arbitrary and is never varied to improve results.
- **Status:** Fixed. Set in the original scaffold (`6bd71a1`), before any result existed.

---

## Phase 1 — Problem definition

All Phase 1 decisions were made in the original scaffold or feature commits (`6bd71a1`–`648b780`,
22–24 Oct 2026). At that point the metric functions were unimplemented, so no model result could
have been seen.

### D-010 · Target
- **Decision:** `is_delayed = 1` if `arr_delay >= 15` minutes, else 0.
- **Reason:** US BTS on-time convention (a flight is on time if it arrives < 15 min late).
- **Status:** Fixed.

### D-011 · Flights without an arrival-delay value
- **Decision:** Exclude cancelled flights and flights that departed with no arrival record
  (`cancelled_policy: exclude`).
- **Options considered:** `exclude` vs `delayed` (treat any disruption as delayed).
- **Reason:** The model predicts arrival delay *given that the flight operates and arrives*.
  Cancellations and diversions are a different prediction problem (`docs/leakage_policy.md` §1).
- **Status:** Fixed. Scope limitation to state in the report.

### D-012 · Decision time
- **Decision:** End of day D−1 (`decision_time.mode: d_minus_1`). Features for a flight on date D
  may use the published schedule and outcomes known before D.
- **Option considered:** `sched_dep_minus_2h` (recorded in the original config).
- **Reason:**
  - *Use:* predictions support next-day operational planning (crew, spare aircraft and gate
    buffers), which happens the evening before.
  - *Data:* end of D−1 is the latest cutoff at which every feature can be computed for a whole
    day's schedule in one batch. All of D's scheduled flights are published, and outcomes up to
    D−1 are known.
  - *Why not later:* a cutoff such as 2 hours before departure would need same-day realtime data
    that this dataset cannot validate.
- **Link to later decisions:** Planners act on every flagged flight, so false alarms have a
  cost. This is the context for the operating precision target (D-015), which is decided with
  the experiment plan.
- **Status:** Fixed. Reason recorded 2026-10-25.

### D-013 · Chronological split
- **Decision:** train 2013-01-01–08-31, validation 09-01–10-31, test 11-01–12-31. No shuffling.
- **Origin:** Present in the first commit (`6bd71a1`).
- **Known issue:** Validation has a much lower delay rate (15.7%) than train (26.4%). Thresholds
  and calibration fitted on validation may transfer poorly.
- **Decision on the issue:** Keep the dates. Changing them after the earlier iteration saw test
  results would itself be a test-informed decision. Report the shift as a limitation.
- **Status:** Fixed.

### D-014 · Primary metric
- **Decision:** PR-AUC is primary. ROC-AUC, Brier score, ECE and recall at the operating precision
  are secondary.
- **Reason:** About one flight in four is delayed, so the classes are imbalanced. PR-AUC focuses
  on the positive (delayed) class. README states this from the scaffold onward.
- **Status:** Fixed.

### D-015 · Operating precision target
- **Current value:** `evaluation.operational_precision: 0.60`.
- **Origin:** Scaffold value marked "TODO: justify". No recorded reason.
- **Reason:** At least 60% of flagged flights should truly be delayed, because operations
  planners act on every flag (D-012) and false alarms waste schedule buffers.
- **Status:** **Assumption.** The value 60% is not derived from real cost data; no costs of false
  alarms or missed delays are available. It is fixed before any calibration or threshold result
  and is not changed based on model results. Recorded 2026-10-25.

---

## Phase 2 — Exploratory analysis on the training period

### D-101 · EDA re-run on Jan–Aug only
- **Decision:** Every EDA statistic that informs feature design uses the training period only
  (2013-01-01 to 2013-08-31, 217,727 labelled flights, 26.4% delayed). Validation and test
  outcomes are not used. This replaces the earlier full-year EDA.
- **Where:** `notebooks/02_eda.ipynb`. It loads through `load_dev_raw` and filters to the
  `split.train` dates. It adds two training-only checks the earlier version lacked: congestion
  within the same departure hour (§10) and holiday proximity (§11). Findings, and a comparison
  with the full-year numbers, are in the notebook's final cell.
- **Evidence by feature group:**

| Feature group | Jan–Aug evidence | Verdict |
|---|---|---|
| Hour: `sched_dep_hour_frac`, `dep_hour_sin/cos`, `dep_bucket` | Rate rises from 10.8% (05:00) to 40.5% (21:00) | **Supported** |
| Weekday: `dow`, `dow_sin/cos`, `is_weekend` | Thu 31.0%, Sat 18.5%, Sun 23.0% | **Supported** |
| Categoricals: carrier, origin, dest, route | EV 35.3% vs AA 20.3%; EWR 28.5% vs LGA 24.4%; destinations 20.4–41.4% | **Supported** |
| Crosses: `carrier_origin` (and route-level) | 9E EWR 19.1% vs JFK 29.5%; ORD JFK 33.9% vs LGA 22.9% | **Supported** |
| Crosses: `origin_dep_bucket`, `carrier_dep_bucket` | Not examined directly | **Not examined** |
| History, 1-day window | Daily rate lag-1 autocorrelation 0.356 | **Supported** |
| History, 7- and 28-day windows | Lag-2 0.041, lag-7 0.021 (full year showed 0.13 / 0.11). Remaining rationale: monthly level shifts within Jan–Aug (22.7–32.2%) and per-key base-rate differences | **Weak / indirect** |
| Congestion (4 features) | Overall flat (ρ 0.014), confounded with hour. Within the same hour, busier slots more delayed in 15 of 18 hours (weighted ρ 0.05; evening +14 to +18 points) | **Weak** |
| Schedule: distance, block time, etc. | Distance weak and non-monotonic (23.0–29.7% across deciles) | **Weak** |
| Holiday proximity (3 features) | Five holidays in the period; holiday dates 18.0% vs 27.6% far from holidays; non-monotonic between | **Inconclusive** |
| Month / day-of-year identity | Already excluded (validation and test months are unseen in training). Within Jan–Aug, monthly rates shift between 22.7% and 32.2% | Exclusion consistent |

- **What changed relative to the full-year EDA:** Only the multi-day persistence result. Lags 2
  and 7 were positive with all twelve months included and are near zero on Jan–Aug. Every other
  pattern holds with similar magnitude.
- **Use of these results:** The verdicts are inputs to D-102. Any feature group kept without
  training-data support must have a written prior reason (D-103), and none is to be added,
  dropped or tuned on validation or test scores.
- **Status:** Fixed. Date: 2026-10-25.

---

## Phase 3 — Feature set and fixed settings

`notebooks/03_feature_engineering.ipynb` was re-run on the development data. Its outcome-based
audits (taxi times, cancellations, label timing) now use the training period only; audit A3 still
compares train with validation by design. No Nov–Dec value appears in any output.

### D-102 · Keep all 59 features
- **Decision:** Keep every feature group: calendar (10), schedule/route (6), categorical (10),
  congestion (4) and historical (29). `test_feature_set_is_pinned` fixes the set, so any later
  change fails until this log and the test are updated together.
- **Options considered:**
  1. Keep all 59 features, with a written reason for each group kept on weak evidence.
  2. Drop the groups D-101 rated weak, inconclusive or not examined.
- **Reasons for option 1:**
  - D-101 measures one feature at a time. Weak single-feature association does not show a
    feature is useless alongside others (congestion is flat overall but positive within the
    same hour).
  - The 7- and 28-day history windows are the model's main way to follow level shifts. Month
    identity is excluded (audit A3). Validation has a much lower delay rate than training
    (15.7% vs 26.4%), and the 28-day origin rate tracks that drop (0.24 → 0.13–0.18).
  - Both planned models cope with uninformative features: L2-regularised Logistic Regression and
    XGBoost.
  - Dropping is not reversible without the choice looking result-driven. Keeping all 59 and
    testing the weak groups on validation under a plan written in advance is.
- **Reasons per group kept on weak or no direct evidence:**

| Group | Reason it is kept |
|---|---|
| 7- and 28-day history | Follow per-key and seasonal level shifts (monthly rates 22.7–32.2% within Jan–Aug) without month identity |
| Congestion | Within the same hour, busier slots are more delayed in 15 of 18 hours (D-101); schedule-only, known at D−1 |
| Distance and schedule numerics | Schedule-only and known at D−1; `sched_block_min` validated (0% negative taxi times) |
| Holiday proximity | Holiday travel peaks are a standard aviation effect; only five holidays in training, so the data cannot settle it |
| Origin × departure-bucket, carrier × departure-bucket | Let an additive Logistic Regression represent hour effects that differ by airport and carrier |

- **Follow-up:** D-201 (experiment plan) will specify a validation-only feature-group check,
  written before any training. It must not be decided on test.
- **Status:** Fixed. Date: 2026-10-25.

### D-103 · Fixed feature settings
These values are not tuned on any model result. The evidence below is from the training period
(Jan–Aug).

| Setting | Value | Reason | Training-period evidence |
|---|---|---|---|
| History windows | 1, 7, 28 days | 1 day = the previous day. 7 and 28 are whole weeks, so each window contains every weekday equally and the weekday mix cannot bias the rate. 28 days ≈ one month, matching the scale of monthly level shifts | Lag-1 autocorrelation 0.356; monthly rates 22.7–32.2% |
| Prior strength *m* | 20 | Shrinks sparse keys toward the global past rate: a key needs 20 known flights before its own rate gets half the weight | Route 1-day windows have median n = 9 (89% below 20); carrier and origin rarely below 20 |
| Category minimum | 50 training rows | Below about 50 rows, a level's delay rate is too uncertain (95% half-width about ±12 points at the 26.4% base rate) | Collapsed levels cover at most 0.45% of training rows per column |
| Departure buckets | 0, 5, 9, 14, 18, 22 h | Conventional time-of-day boundaries (night, early morning, midday, afternoon, evening) | Rates rise across buckets: early morning 14.0%, midday 21.1%, afternoon 32.8%, evening 38.0%; night 37.4% (n = 2,482) |

- **Status:** Fixed. Date: 2026-10-25.

### D-104 · 240-minute fallback buffer
- **Decision:** Keep 240 minutes.
- **Evidence (training period only):** the longest taxi time (actual gate-to-gate minus air time,
  on a UTC basis) is 210 minutes. 240 exceeds it, so a fallback estimate cannot place an outcome
  before the flight could have arrived. This replaces the earlier justification, which used the
  full-year maximum (235 min).
- **Practical effect:** None in this data. Every labelled flight's arrival can be placed, so the
  fallback is used on 0 rows.
- **Status:** Fixed. Date: 2026-10-25.

### D-105 · 30-minute reporting buffer
- **Decision:** A past flight's outcome becomes usable 30 minutes after its actual arrival.
- **Status:** **Assumption.** The dataset has no information on reporting latency. The value
  assumes a near-real-time flight-status feed. It is not compared across values by model score.
  Public BTS files are released monthly, so a system relying on them could not compute these
  windows. Report this as a limitation.
- **Date:** 2026-10-25.

### D-106 · Data-quality notebook on development data
- **Decision:** `notebooks/01_data_quality.ipynb` runs on the development file (Jan–Oct) through
  `load_dev_raw`. Its saved outputs previously included full-year statistics, among them the
  test-period delay rate and Nov–Dec cancellation rates. These are replaced. The split-balance
  table shows train and validation only.
- **Scope:** The notebook checks data quality (schema, missing values, duplicates, invalid values,
  label construction). It informs no feature or model choice. Using validation-period rows here
  is therefore allowed; the test holdout is not read.
- **Result:** All quality conclusions hold on Jan–Oct: 0 error-level violations, 0 duplicates,
  and `tailnum` is missing only for cancelled flights.
- **Status:** Fixed. Date: 2026-10-25.

### D-107 · Documentation free of test-period statistics
- **Decision:** No document states a statistic computed on Nov–Dec data.
- **Changes:**
  - `docs/data_dictionary.md`: every count and rate is recomputed on the Jan–Oct development
    file. The test-period delay rate (previously stated) is removed.
  - `docs/feature_dictionary.md` and `docs/leakage_policy.md`: two exclusion reasons that cited
    test values are restated without them. Cumulative counts and day-of-year grow with the date,
    so any later period is out of range by construction. Audit numbers are replaced with
    training-period values: taxi maximum 210 min (was 235), 19 early labels (was 23), 25.6% of
    slots with a later-cancelled flight (was 22.7%).
  - `README.md`: problem definition and data pipeline updated; training lines unchanged.
- **Code hygiene:** `features/calendar.py` uses `day_of_week` / `day_of_year` instead of the
  deprecated pandas names. A rebuild produced identical train and validation matrices and a
  byte-identical encoder and feature spec.
- **Status:** Fixed. Date: 2026-10-25.

---

## Feature engineering frozen

Feature engineering ends here. Tag `features-frozen` marks the commit. At that point:
- the feature set is 59 features, pinned by `test_feature_set_is_pinned`;
- every feature decision (D-101 to D-107) rests on training-period evidence or a written reason;
- the Nov–Dec holdout has not been read since D-002;
- no model has been trained, and no model, metric or calibration code exists on this branch.

Any later change to features must be a new, logged decision made without test data.

---

## Pending decisions

None. All decisions from D-000 to D-205 are recorded.

---

## Phase 4 — ML experiment plan

### D-201 · Pre-registered modelling protocol

**Status:** Approved on 2026-10-25, as written in commit bf7a51c.

**Chronological data protocol**

- January 1–July 31, 2013: train candidate models.
- August 1–31, 2013: tune hyperparameters, XGBoost early stopping,
  select the winner, and generate validation-only comparisons.
- After selection: refit the winning model on January 1–August 31.
- September–October are not used in D-201.
- November–December remain sealed for the separately authorized final test.

**Features and preprocessing**

- Use exactly 59 frozen inputs: 49 numeric and 10 categorical.
- Fit categorical encoder and numeric imputation using January–July only
  for tuning experiments.
- Logistic Regression also fits StandardScaler and OneHotEncoder on
  January–July only.
- During final January–August retraining, refit all learned preprocessing
  from scratch using January–August.
- No feature additions or removals after inspecting August results.
- Random seed: 42.

**Logistic Regression hyperparameter grid**

- L2 regularization.
- C: {0.01, 0.1, 1, 10}.
- class_weight: {None, balanced}.
- max_iter: 2000.
- Eight combinations.

**XGBoost hyperparameter grid**

- max_depth: {3, 5}.
- learning_rate: {0.03, 0.10}.
- Four combinations.
- Maximum n_estimators: 1000.
- early_stopping_rounds: 50 on August.
- subsample: 0.8.
- colsample_bytree: 0.8.
- scale_pos_weight = 1.0 (fixed, not tuned).
- objective: binary:logistic.
- eval_metric: aucpr.
- tree_method: hist.
- enable_categorical: True.
- Record best_iteration and the effective number of trees for every run.

**Model selection**

- Rank candidates using August average precision (PR-AUC).
- Select the highest scoring candidate.
- In an exact tie, prefer Logistic Regression over XGBoost;
  otherwise use the fixed grid order.
- Log all settings, August PR-AUC and ROC-AUC.
- August results are tuning scores, not final held-out performance.

**Final model**

- After tuning, retrain the winning model once on January–August.
- Refit encoder, numeric imputer, scaler and one-hot preprocessing
  as applicable on January–August.
- Logistic Regression uses its selected C and class_weight.
- XGBoost uses selected max_depth and learning_rate, and fixes
  n_estimators = best_iteration + 1 from August tuning.
- Disable XGBoost early stopping for final retraining.
- Save and hand over this January–August retrained model.
- Do not report August predictions from this retrained model as
  independent validation results.

**Feature importance — reporting only**

- Report gain-based feature importance for the tuned XGBoost model.
- Report permutation importance on August using the tuned
  January–July models, with PR-AUC as the scoring metric.
- Do not change the frozen feature set based on these findings.

**Experiment deliverables**

- Record every tuning run and its August PR-AUC and ROC-AUC.
- Produce PR-AUC/ROC-AUC comparison tables and PR/ROC curves
  from January–July candidate model predictions on August.
- Record final selection, August scores, selected hyperparameters,
  and January–August retraining details under D-202.
- Save the final retrained model and preprocessing artifacts.
- Keep the November–December holdout sealed.

## D-202 — Final model selection

Status: Completed after D-201 candidate training.

- Selection dataset: August 2013.
- Candidate fitting dataset: January–July 2013.
- Selection metric: highest August average precision (PR-AUC).
- Winning model: Logistic Regression.
- Logistic Regression PR-AUC: 0.400711.
- Logistic Regression ROC-AUC: 0.675593.
- Best XGBoost PR-AUC: 0.396029.
- Best XGBoost ROC-AUC: 0.677037.
- Selected hyperparameters: `{"C": 0.01, "class_weight": "balanced", "max_iter": 2000}`.
- Final model retrained on January–August 2013.
- FeatureEncoder and model preprocessing refitted on January–August.
- August metrics are from January–July-fitted candidates, not the final refit.
- Fixed feature set: 49 numeric + 10 categorical.
- XGBoost gain and permutation importance are reporting-only.
- Permutation importance: 2000 August examples, seed 42, 2 repeats.
- No calibration or operating-threshold selection in this workflow.
- November–December sealed holdout not accessed.
- Training environment: Python 3.11.17, scikit-learn 1.9.1, XGBoost 3.2.0.
- The feature-group diagnostic from the first D-201 draft was dropped in the revised plan and was not run.

---

## Phase 5 — Evaluation, calibration and rescheduling

### D-203 · Evaluation plan
- **Status:** Written before any calibration, threshold or test result. Date: 2026-10-25.
- **Owner:** Ganga Raghuwanshi, carrying out the Member 4 role.
- **Model:** `models/d201/final_model.joblib` (D-202 Logistic Regression, trained on
  January–August), used unchanged. No retraining, no feature or hyperparameter change.
- **Inputs:** `data/processed/valid.parquet`. Its encoder is identical to
  `models/d201/final_encoder.json` (checked in the D-202 review), so its 59 feature columns are
  already encoded for the final model. Column order follows `data/processed/feature_spec.json`
  (`numeric`, then `categorical`).

**Data periods (one job each)**

| Period | Use |
|---|---|
| 2013-09-01 to 2013-09-30 | Fit the probability calibration |
| 2013-10-01 to 2013-10-31 | Choose the threshold; apply the rescheduling rule |
| 2013-11-01 to 2013-12-31 | Single test evaluation, after `model-frozen` |
| January–August | Not used (already used by D-201/D-202) |

**Calibration**
- Method: **sigmoid**, fitted on September only with `models.calibration.calibrate`. The
  underlying model is frozen, and one calibrator is fitted on all September rows
  (`tests/test_evaluation_checks.py`).
- Isotonic is also fitted on September and reported for comparison only. It is never selected.
- Needed because the final model uses `class_weight="balanced"`, which pushes raw probabilities
  upward.
- Reported: Brier score, ECE (10 equal-width bins) and a reliability diagram, for raw and
  calibrated probabilities. October is the out-of-sample check for the calibrator; September is
  reported as in-sample reference.

**Threshold**
- On October calibrated probabilities, `models.operating_point.choose_threshold` with
  `min_precision = 0.60` (D-015): the highest recall with precision at least 0.60; ties go to
  higher precision, then higher threshold. A flight is flagged when its probability is at or
  above the threshold.
- If no threshold reaches 0.60 precision, the result is recorded as **infeasible**. The target
  is not lowered, no threshold is chosen, and threshold-based test metrics are reported as not
  available.

**Rescheduling rule (offline what-if)**
- Flag-and-buffer: flights whose calibrated delay probability is at or above the October
  threshold are flagged for an extra schedule buffer.
- Reported: number and share of flights flagged, share of flagged flights actually delayed
  (precision), and share of delayed flights flagged (recall).
- The data cannot show what a buffer would have changed, so no buffer size or delay reduction
  is estimated.

**Metrics** (October, then the test period): PR-AUC, ROC-AUC, Brier score, ECE (10 bins),
reliability diagram, and at the threshold: precision, recall and flights flagged.

**Test evaluation (once)**
1. D-204 is committed and the owner tags that commit `model-frozen`.
2. `scripts/evaluate.py` loads the full-year file with `load_raw`, builds features for the
   whole year (so test flights get their correct history), keeps `split == "test"` rows, encodes
   them with `models/d201/final_encoder.json`, and applies the calibrated model, threshold and
   rescheduling rule from D-204.
3. In the same commit, `scripts/evaluate.py` is added to `SEALED_READERS` in
   `tests/test_holdout_seal.py`.
4. The script is run once. Results are committed immediately as D-205, exactly as produced.
   Nothing is changed or re-run afterwards.

**Limitations**
- October chooses the threshold, so October operating metrics are optimistic.
- Delay rates differ between months (training 26.4%, validation 15.7%), so a threshold chosen
  on October may transfer poorly to later months.
- The 60% precision target is an assumption (D-015).

### D-204 · Calibration, threshold and rescheduling results
- **Status:** Completed under D-203 (plan commit `968067b`). Date: 2026-10-25.
- **Script:** `scripts/calibrate_threshold.py`, run once. It refuses to overwrite its outputs.
- **Data:** September (27,010 flights) for calibration; October (28,618 flights) for the
  threshold. No January–August or November–December data used.
- **Outputs:** `models/d203/calibrated_model.joblib` (sigmoid on the unchanged D-202 model),
  `models/d203/operating_point.json`, `reports/results/d203_calibration.csv`,
  `reports/figures/d203_reliability_september.png`, `reports/figures/d203_reliability_october.png`.

**Calibration (sigmoid, fitted on September)**

| Period | Probabilities | PR-AUC | ROC-AUC | Brier | ECE (10 bins) | Mean probability | Delay rate |
|---|---|---|---|---|---|---|---|
| September (in-sample) | raw | 0.2457 | 0.6689 | 0.1790 | 0.2402 | 0.3774 | 0.1372 |
| September (in-sample) | sigmoid | 0.2457 | 0.6689 | 0.1129 | 0.0095 | 0.1372 | 0.1372 |
| October (out-of-sample) | raw | 0.3248 | 0.6861 | 0.1917 | 0.2270 | 0.4026 | 0.1756 |
| October (out-of-sample) | sigmoid | 0.3248 | 0.6861 | 0.1360 | 0.0274 | 0.1483 | 0.1756 |
| October (comparison only) | isotonic | 0.3160 | 0.6856 | 0.1371 | 0.0293 | 0.1469 | 0.1756 |

- Raw probabilities are far too high (October mean 0.40 vs a 0.18 delay rate), as expected from
  `class_weight="balanced"`. Sigmoid calibration cuts October ECE from 0.227 to 0.027 and Brier
  from 0.192 to 0.136. It does not change the ranking (PR-AUC and ROC-AUC unchanged).

**Threshold (October, precision ≥ 0.60): infeasible**
- No threshold reaches 0.60 precision on October. As D-203 requires, the target is not lowered
  and no threshold is set.
- Diagnostic only, not a decision: the highest October precision with at least 20 delayed
  flights flagged is 0.513 (158 flights flagged, recall 0.016). The highest calibrated
  probability of any October flight is 0.525.

**Rescheduling rule:** not applied, because there is no feasible threshold.

**Consequences for the test evaluation (D-205)**
- The calibrated model and the infeasible operating point are frozen as recorded here.
- Test metrics: PR-AUC, ROC-AUC, Brier score, ECE (10 bins) and a reliability diagram for raw and
  calibrated probabilities.
- Threshold-based test metrics and the rescheduling rule are reported as not available.

### D-205 · Single test evaluation (November–December)
- **Status:** Completed once, 2026-10-25. Final; nothing is changed or re-run after this result.
- **Run:** `scripts/evaluate.py` (commit `041334b`), against the frozen state tagged
  `model-frozen` (commit `e559e20`). The script checked that the frozen model, encoder,
  calibrated model and operating point were byte-identical to the tagged versions, that no test
  results existed, and that the full-year file matched its D-002 hash.
- **Data:** 53,991 labelled flights, 2013-11-01 to 2013-12-31. Features were built over the whole
  year (past-only history), then only test rows were scored. Nothing was fitted.
- **Outputs:** `reports/results/test_metrics.json`, `reports/results/test_metrics.csv`,
  `reports/figures/test_reliability.png`.

**Results (exactly as produced)**

| Probabilities | Flights | Delay rate | Mean probability | PR-AUC | ROC-AUC | Brier | ECE (10 bins) |
|---|---|---|---|---|---|---|---|
| raw | 53,991 | 0.2583 | 0.4614 | 0.3979 | 0.6640 | 0.2264 | 0.2039 |
| sigmoid (frozen) | 53,991 | 0.2583 | 0.1817 | 0.3979 | 0.6640 | 0.1852 | 0.0768 |

- **Threshold metrics:** not available. No threshold reached 0.60 precision on October (D-204).
- **Rescheduling rule:** not applied, for the same reason.

**Reading of the results**
- **Ranking:** PR-AUC 0.398 against a 0.258 delay rate, about 1.5 times a random ranking.
  ROC-AUC 0.664. These are in line with validation (PR-AUC 1.8 times its delay rate on
  September–October, ROC-AUC 0.67–0.69), so the model generalised to the test period without a
  collapse.
- **Calibration:** the frozen sigmoid calibration cuts ECE from 0.204 to 0.077 and Brier from
  0.226 to 0.185. Calibrated probabilities are too low on average (0.18 against a 0.26 delay
  rate), because calibration was fitted on September, the calmest month (13.7% delayed). The
  shift between validation and test delay rates, recorded as a limitation since D-013, is the
  cause.
- **Operating target:** the 60% precision target (D-015) is not reachable with the information
  available at the end of D−1 in this dataset.

**Limitations to report**
- The dataset (provided by the course) has no weather or same-day information at decision time,
  which limits how confident any flight-level prediction can be.
- Validation (September–October) is calmer than the test period, so probabilities calibrated on
  it under-predict in November–December.
- An earlier iteration of this project evaluated on the Nov–Dec holdout (D-000). Every model,
  feature, calibration and threshold decision reported here was made again without those results.

## Phase 2 — Deep learning

### D-301 · Phase 2 protocol
- **Status:** Adopted 2026-11-12, before any Phase 2 model was selected on it.
- **Decision:** Phase 2 models follow the D-201 protocol. Fit on January–July; choose epochs and
  hyperparameters on August PR-AUC; refit on January–August with the chosen settings (fixed epoch
  count, no early stopping); then score September–October once. Encoders, scalers and vocabularies
  are fitted on the training rows of each stage only. November–December stays sealed (D-002) and
  is not used in Phase 2.
- **Comparison rule:** every Phase 2 model is compared with the Phase 1 Logistic Regression (D-202
  final model, also trained on January–August) on the same September–October flights, with PR-AUC
  as the primary metric and ROC-AUC, Brier and ECE alongside.
- **Why:** the first MLP run chose its epoch on September–October and reported the same period,
  which overstates the result, and it compared that score with Phase 1's August scores, a different
  period with a different delay rate (24.6% against 15.7%).

### D-302 · Member 1 simple MLP baseline
- **Run:** `scripts/train_mlp_baseline.py`, CPU, torch 2.15.0, seed 42. Outputs in
  `models/phase2/member1/` and `reports/{results,figures}/phase2_member1/`.
- **Model:** 1,015 inputs (49 standardised numeric + 966 one-hot) → 128 → 64 → 1, ReLU, dropout
  0.3/0.2, Adam lr 0.001, batch 256, unweighted BCE.
- **August selection:** best epoch 1 (PR-AUC 0.3796; Phase 1 LR 0.4007, XGBoost 0.3960). August
  loss rises from epoch 2, so the network overfits after one pass.

| September–October (55,628 flights, delay rate 0.157) | PR-AUC | ROC-AUC | Brier | ECE |
|---|---|---|---|---|
| Logistic Regression (D-202, raw) | 0.2896 | 0.6800 | 0.1855 | 0.2334 |
| Simple MLP (raw) | 0.2773 | 0.6679 | 0.1344 | 0.0856 |

- **Reading:** the MLP ranks slightly worse than Logistic Regression. Its raw probabilities are
  better calibrated than the class-weighted Logistic Regression's, but still too high (mean 0.243).
  No usable threshold reaches 0.60 precision.


### D-303 · Member 2 GRU: pre-registered design
- **Status:** Written 2026-11-12, before any GRU or control model was trained.
- **Question:** do ordered daily sequences of past delays, read by a compact GRU, improve
  next-day prediction over the Phase 1 Logistic Regression and the Member 1 MLP, and is any gain
  due to reading the days in order (recurrence) rather than to the extra inputs?
- **Sequences:** for a flight on day D, the last L days D-L ... D-1, oldest first. A past flight
  counts on the day its outcome became known (`outcome_known_date`, D-104/D-105), so a D-1
  flight landing after 23:30 New York time is not in D's sequence. Ten values per day: delay
  rate for the whole system, the flight's origin, carrier and destination (each smoothed towards
  the global past rate with prior strength 20, as in Phase 1); log(1 + flights) for system,
  origin and carrier; a has-data flag for the carrier; and the day's weekday as sine/cosine.
  Only past delayed/not-delayed labels are used (no delay minutes, no cancellations: tested on
  January-August, they add at most 0.006 R² for the next day's rate). Days before 1 January are
  padding (flag 0). Channel scaling is fitted on each stage's training flights only.
- **Models** (identical static branch and head; only the sequence encoder differs):
  - *GRU:* 1-layer GRU over the L x 10 sequence (hidden size h), final state.
  - *Control (no recurrence):* the same L x 10 values flattened -> Linear(h) -> ReLU.
  - Both join the flight's 59 Phase 1 features (Member 1's `MLPPreprocessor`: scaled numeric +
    one-hot) -> Linear(64) -> ReLU -> Dropout(0.2) -> Linear(1).
  - *Member 1 MLP:* unchanged (D-302), re-run with the seeds below.
- **Protocol:** D-301. Adam lr 0.001, batch 256, unweighted BCE, max 30 epochs, early stopping
  with patience 5 on August PR-AUC, refit January-August for the chosen epoch count, CPU.
- **Grid (GRU only, seed 42):** L in {7, 14, 28} x h in {16, 32}. Highest August PR-AUC wins;
  ties within 0.001 go to the smaller (L, then h). The control uses the selected L and h.
- **Seeds:** the selected GRU, the control and the Member 1 MLP are each run with seeds 42-46.
  Each seed selects its own epoch on August and is refit January-August. Results are reported as
  mean and standard deviation over seeds.
- **Evaluation:** September-October once, with the Phase 1 Logistic Regression (D-202, raw) on
  the same flights. PR-AUC (primary), ROC-AUC, Brier, ECE; September and October also separately.
  Differences in PR-AUC (GRU - LR, GRU - control, GRU - MLP) get 95% intervals from a bootstrap
  that resamples whole days (1,000 resamples, seed 42), averaging each difference over the 5
  seeds.
- **Diagnostic (not used for selection):** the selected GRU without the 29 Phase 1 historical
  features, 5 seeds, to see whether the sequence alone recovers that information.
- **Reading rule:** a model "improves" on another only if the 95% interval of the PR-AUC
  difference excludes 0.

### D-304 · Member 2 GRU results
- **Run:** `scripts/train_gru.py` exactly as pre-registered in D-303, CPU, torch 2.15.0, about
  9 minutes. Outputs in `models/phase2/member2/` and `reports/{results,figures}/phase2_member2/`.
  Details: `docs/phase2_member2_gru.md`.
- **Grid (August PR-AUC, seed 42):** all six settings within 0.004 (0.368-0.372). Selected
  L = 14 days, hidden size 16 (0.3719). Every GRU seed chose epoch 1 or 2.
- **Checks against earlier results:** Member 1 MLP seed 42 reproduces D-302 exactly (0.2773);
  the Logistic Regression reproduces 0.2896.

| September-October, mean ± sd over seeds 42-46 | PR-AUC | ROC-AUC | Brier | ECE |
|---|---|---|---|---|
| GRU (L = 14, h = 16) | 0.2946 ± 0.0055 | 0.6743 | 0.1360 | 0.0863 |
| Logistic Regression (D-202, raw) | 0.2896 | 0.6800 | 0.1855 | 0.2334 |
| GRU without the 29 history features | 0.2885 ± 0.0064 | 0.6748 | 0.1347 | 0.0840 |
| Member 1 MLP | 0.2839 ± 0.0043 | 0.6719 | 0.1333 | 0.0757 |
| Control (same inputs, no recurrence) | 0.2553 ± 0.0214 | 0.6433 | 0.1458 | 0.0820 |

| PR-AUC difference (day bootstrap, 95%) | Estimate | Interval | Counts? |
|---|---|---|---|
| GRU - LR | +0.0050 | [-0.0137, +0.0238] | No |
| GRU - control | +0.0393 | [+0.0087, +0.0718] | Yes |
| GRU - MLP | +0.0106 | [-0.0004, +0.0216] | No |
| GRU without history - GRU | -0.0061 | [-0.0220, +0.0078] | No |

- **Reading:** reading the days in order beats feeding the same values in flat, but no Phase 2
  model so far beats the Phase 1 Logistic Regression beyond noise. The sequence alone recovers
  almost all of the 29 history features. Raw GRU probabilities are too high (mean 0.243 vs
  0.157) and need calibration (Member 5).

### D-305 · Member 3 embedding MLP: pre-training experiment plan

- Status: registered before Member 3 training.
- Goal: evaluate learned categorical embeddings with the existing 49 numeric
  and 10 categorical features, following D-301.
- Architecture: concatenate scaled numeric inputs and categorical embeddings;
  two ReLU hidden layers, dropout after each, and one output logit.
- Embedding dimension per category:
  min(cap, ceil(sqrt(training vocabulary size))), minimum 1.
- Tuning grid, seed 42:
  - learning rate: {0.0001, 0.0003}
  - hidden layers: {(64, 32), (128, 64)}
  - embedding dimension cap: {4, 8}
  - Total: 8 configurations.
- Fixed settings: Adam, weight decay 0.0001, dropout 0.2,
  batch size 256, unweighted BCEWithLogitsLoss, CPU,
  maximum 30 epochs, early-stopping patience 5.
- Stage A: fit January-July. Fit categorical vocabularies, rare-category
  collapsing, missing-value imputation and numeric scalers on these rows only.
  Map unknown categories to the reserved unknown/other category.
- Epoch selection: highest August PR-AUC; retain the earliest epoch when
  scores tie. Early stopping uses August PR-AUC only.
- Configuration selection: highest August PR-AUC at its best epoch.
  Exact ties prefer fewer trainable parameters, then lower learning rate,
  then deterministic configuration order.
- Freeze the selected configuration before final runs.
- Final seeds: 42, 43, 44, 45, 46. Each seed selects its epoch on August
  using January-July training and the frozen configuration.
- Stage B: reinitialize each model and refit preprocessing on January-August.
  Train for exactly its August-selected epoch count, without early stopping.
- Evaluate all frozen final models on September-October once at the end.
  Do not use September-October for settings, epochs or model selection.
- Compare on identical flight IDs and labels with saved Member 1 MLP,
  Logistic Regression and GRU predictions from
  reports/results/phase2_member2/sep_oct_predictions.parquet.
- Report PR-AUC, ROC-AUC, Brier and ECE, with mean and standard deviation
  across seeds. Compute paired PR-AUC differences using day_bootstrap
  imported from scripts/train_gru.py, with its 1,000 day resamples.
- Claim improvement only when the difference's 95% interval is above zero.
- Reuse Member 1 and Member 2 utilities through imports.
  Keep their implementation and saved artifacts intact.
- November-December remains sealed and is excluded from this work.
- Save Member 3 code, checkpoints, preprocessing, run histories,
  August grid results and final aligned predictions separately.

### D-306 · Member 3 embedding MLP results

- Completed under the unchanged D-305 grid and selection rule, on CPU with
  Python 3.11.17 and PyTorch 2.15.0. D-305 was committed as `12f13e8` before training.
- The existing holdout-reference test rejected an unnecessary sealed-path name in
  the new script's path comparison. Commit `9e01bfb` replaced it with an explicit
  dedicated-development-path check. The identical August grid was rerun after
  this implementation fix. September–October had not been scored at that point.
- All eight registered settings were evaluated with seed 42. Run03 won on August
  PR-AUC 0.3968296, with learning rate 0.0001, hidden layers (128,64), embedding
  dimension cap 4, and epoch 2. These are tuning results.
- The frozen architecture selected epochs 2,3,2,5,2 for seeds 42,43,44,45,46,
  respectively. Each seed was reinitialised and refit on January–August.
  Vocabularies, numeric medians and scalers were refit using those rows only.
- The final model concatenates 49 numeric inputs and 37 embedding values,
  followed by the 128/64 dense head. It has 23,315 trainable parameters.
- Final model scoring used the same 55,628 September–October flights as Member 2,
  with identical flight IDs, dates and labels. All probabilities are raw.

| Model | PR-AUC mean ± sd | ROC-AUC | Brier | ECE |
| --- | --- | --- | --- | --- |
| Embedding MLP | 0.2762 ± 0.0024 | 0.6753 | 0.1279 | 0.0473 |
| Member 1 MLP | 0.2839 ± 0.0043 | 0.6719 | 0.1333 | 0.0757 |
| Logistic Regression | 0.2896 | 0.6800 | 0.1855 | 0.2334 |
| Member 2 GRU | 0.2946 ± 0.0055 | 0.6743 | 0.1360 | 0.0863 |

| Paired PR-AUC difference | Estimate | 95% day-bootstrap interval |
| --- | --- | --- |
| Embedding MLP − Member 1 MLP | −0.0077 | [−0.0253, +0.0111] |
| Embedding MLP − Logistic Regression | −0.0134 | [−0.0212, −0.0034] |
| Embedding MLP − GRU | −0.0184 | [−0.0387, +0.0020] |

- Intervals use the unchanged `day_bootstrap` from `scripts/train_gru.py`, with
  1,000 whole-day resamples and per-seed PR-AUC averaged within each sample.
  The registered criterion supports lower ranking performance than LR. The
  embedding model has smaller raw-probability Brier and ECE values. Different
  architectures and loss weighting also influence that comparison.
- No settings or epochs changed after the final evaluation. November–December
  was excluded throughout this Member 3 work. Earlier project exposure remains
  disclosed under D-000 and D-301.
- Code, checkpoints, source hashes, histories, predictions and uncertainty results
  are saved under the separate Member 3 paths. Details and reproducible stages
  are in `docs/phase2_member3_embedding_mlp.md`. Member 1 and 2 files retain
  their original contents.
