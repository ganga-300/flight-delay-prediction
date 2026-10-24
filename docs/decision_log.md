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
  results is reused. The modelling stage is redeveloped from scratch on branch `ml-restart`.
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
- **Status:** **Open.** Before Phase 4, record either a cost-based reason (cost of a false alarm vs
  a missed delay) or label it an assumption. It must not be changed based on model results.

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

## Guards for model development

### D-108 · Train/validation-only guards
- **Decision:** Before training starts, the repository enforces the protocol in code and on
  GitHub. Contributors can train on the training period and tune on validation. They cannot
  read the test holdout, change the frozen features or settings, tune outside the plan, or
  evaluate on test more than once without it being blocked or visible.
- **Code guards (tests fail on violation):**

| Guard | Where | Stops |
|---|---|---|
| Train/validation-only data API: `load_dev_split("train" or "valid")`; any other name raises | `src/flight_delay/data/dev_splits.py` | Training or tuning on test |
| Feature list checked against the frozen 59 on every load | `feature_columns()` | Training on a stale or edited feature set |
| One-shot holdout gate: opens only on a commit tagged `model-frozen`, with a clean tree, before any test result exists; checks the sealed file's hash | `src/flight_delay/data/holdout.py` | Early, repeated or uncommitted test evaluation |
| Code scans: sealed-file references, test-split selection (`== "test"`, `["test"]`, Nov–Dec dates, `month >= 11`) and direct file reading are allowed only in named, owner-reviewed files | `tests/test_holdout_seal.py` | Reaching the holdout by another route |
| Config lock: seed, target, decision time, split dates, feature settings and data paths | `tests/test_guards.py` | Quietly changing the split or features |
| Pinned feature set (59 features, group sizes, settings) | `tests/test_features.py` | Dropping or adding features |
| Experiment plan: runs are refused while the plan is a draft, and only grid points of the approved plan are accepted. Only `train_`/`valid_` metrics may be recorded | `configs/experiment_plan.yaml`, `src/flight_delay/experiments.py` | Tuning outside the plan; recording test metrics |
| Run records: every committed record must match the current plan's hash, come from a clean commit in this history, and use the frozen feature count | `tests/test_experiments.py` | Hidden or after-the-fact experiments |

- **GitHub guards:**
  - `.github/workflows/tests.yml` runs the full suite on every pull request and every push to
    `ml-restart`. It builds the dev file and features from scratch first.
  - `.github/CODEOWNERS` requires owner approval for changes to the guards, the frozen data and
    feature pipeline, configs (including the experiment plan), the decision log, run records,
    `scripts/evaluate.py` and test results.
  - Both take effect only once the owner enables branch protection on `ml-restart`: pull
    requests required, the `tests` check required, review from Code Owners required, no force
    pushes. Branch protection on a private repository needs a paid GitHub plan.
- **Verified:** 51 tests pass. Each code scan catches a planted violation. The holdout gate and
  run records were exercised in a throwaway git repository (closed without the tag, closed with
  uncommitted code, opens once, closed after results are committed). A fresh copy of the
  repository builds and passes the suite from scratch.
- **Limits:**
  - The full-year file `data/raw/flights.csv` is still in the repository and its history. The
    guards stop code in the repository from reading it, not a person opening it by hand.
  - The dataset is public, so nothing can stop someone fetching Nov–Dec outside the
    repository.
  - The tests and CODEOWNERS make deliberate misuse visible and blocked at review; they cannot
    prevent local experiments that are never committed.
  - Trained models are not committed (`models/` is git-ignored). The run records are the
    evidence of what was trained.
- **Status:** Fixed. Date: 2026-10-25.

---

## Pending decisions (to be logged in order)

| ID | Phase | Decision | Evidence allowed |
|---|---|---|---|
| D-201 | 4 | Experiment plan, written by Member 3 in `configs/experiment_plan.yaml` (status `approved`) and in this log before any training: models, hyperparameter grid, selection rule, and the validation-only feature-group check from D-102 | Written before any training |
