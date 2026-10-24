# Leakage Policy

The model must not use post-departure information when making pre-departure predictions.
Every rule below is enforced in code and covered by tests (`tests/test_features.py`,
`tests/test_leakage.py`, `tests/test_split.py`).

## 1. Prediction target and scope
- `is_delayed = 1` if `arr_delay >= 15` minutes (BTS convention), else 0.
- Flights without an observed `arr_delay` (cancelled, or departed with no arrival record) are
  excluded (`target.cancelled_policy: exclude`). The model predicts arrival delay ≥ 15 min **for
  flights with an observed arrival-delay label**. It does not predict cancellations or diversions.

## 2. Decision time
`decision_time.mode: d_minus_1` — features for a flight on date **D** are computed at the
**end of day D−1**. Allowed inputs:
- the published schedule, including all flights scheduled on D;
- outcomes of earlier flights that were **known before D**.

Same-day realised information (for example, morning delays on D) is not used.

## 3. Rules
1. **No post-departure columns as features.** `dep_time`, `dep_delay`, `arr_time`, `arr_delay`,
   and `air_time` (`features/leakage.py`) never enter the feature matrix. `tailnum` and `id` are
   also excluded: `tailnum` is missing only for cancelled flights, and `id` is just a row number.
   `build_feature_table` keeps only an explicit allow-list and calls `assert_no_leakage`.
2. **Historical statistics use outcomes known strictly before D.** A past flight contributes
   only from the date its outcome is known: its **actual arrival converted to New York time**
   (destination IANA time zone, DST-aware, arrival date resolved) **+ 30 min**. An arrival after
   midnight on D−1 therefore counts only from D+1. `dep_delay`, `air_time` and `arr_time` are
   used only to *time* past labels, never as features of the flight being predicted. The current
   flight and every flight on date D are therefore excluded from their own history. The
   earlier estimate (departure + delay + air time + 120 min) ignored taxi times, which reach
   210 min in the training period, and counted 19 training-period labels a day early. It is now
   only a fallback for airports without a time zone, with a 240-min buffer that exceeds the
   longest training-period taxi time (D-104). The fallback is used on 0 rows in this data.
3. **Insufficient or unseen history** shrinks toward the global past rate:
   `(k + m·p0) / (n + m)`, where `m = history_prior_strength`. On the very first date (no past data),
   rates are NaN and are imputed with the train median.
4. **Congestion uses the full raw schedule** (origin × date × scheduled hour, including
   flights later cancelled or without labels). It is computed before the missing-label filter,
   and never uses realised delays.
5. **Learned transformations are fitted on the training split only**
   (`features/encoding.py::FeatureEncoder`). This covers rare-level collapsing (`categorical_min_count`),
   mapping unseen levels to `__OTHER__`, and median imputation. The fitted encoder is saved to
   `data/processed/encoder.json` and applied unchanged to validation and test.
6. **The split is chronological.** It uses the inclusive date ranges in `configs/config.yaml`,
   assigned by `data/split.py`, which rejects overlapping or out-of-order ranges.
7. **Features that act as time indices are excluded.** These are raw day-of-year and cumulative
   history counts: they grow steadily with the date, so any later period falls outside the
   training range by construction.
8. **No month-identity features.** Raw and cyclic month/day-of-year encodings are excluded
   because training (Jan–Aug) never contains the validation/test months. Seasonality comes
   only from past-only history windows. Evidence is in notebook 03, audit A3, which uses train
   and validation only.
9. **Durations use a common time basis.** `sched_block_min` converts origin- and destination-local
   scheduled times to UTC with static airport time zones (`features/airport_timezones.py`).
10. **The test holdout is sealed during development.** EDA, feature work, training and model
    selection read only the Jan–Oct file (`paths.raw`), loaded through `load_dev_raw`, which
    rejects any row after 2013-10-31. The full-year file (`paths.raw_sealed`) is read once, by the
    final test evaluation after the model is frozen (`docs/decision_log.md`, D-002).

## 4. Verification
| Check | Where |
|---|---|
| Rebuilding from data truncated at a cutoff reproduces earlier features exactly | `test_real_data_future_rows_cannot_change_earlier_features`, notebook 03 §2 |
| Flipping labels on and after a date leaves history up to that date unchanged | `test_changing_same_day_and_future_labels_does_not_change_history` |
| Late-arriving D−1 flights are not used on D | `test_late_arriving_previous_day_flight_is_not_used_next_day` |
| Long-taxi flights counted only after actual arrival; no label before its NY arrival date (real data) | `test_long_taxi_flight_is_known_only_after_actual_arrival`, `test_real_data_outcomes_never_known_before_arrival` |
| Time zones: no negative taxi time on real data; DST/overnight/Hawaii block times | `test_real_data_time_basis_audit`, `test_scheduled_block_uses_common_time_basis` |
| History count on D equals outcomes known before D (real data) | `test_real_data_history_never_uses_same_or_later_dates` |
| Encoder unaffected by validation/test rows; unseen levels → `__OTHER__` | `test_encoder_is_fitted_on_train_only` |
| Congestion includes unlabelled rows | `test_congestion_counts_use_full_raw_schedule` |
| No post-departure, target or ID columns in features | `test_feature_matrix_has_no_post_departure_or_target_columns` |
| Development data holds no holdout rows; only allowed files reference the sealed data | `tests/test_holdout_seal.py` |

## 5. Limitations that cannot be resolved with this dataset
- **Schedule as known at D−1:** the file holds the final schedule as operated or cancelled, with
  no publication or amendment timestamps. Late schedule changes, and flights missing from the
  file, cannot be detected.
- **Congestion includes cancellations that may have been announced before D−1** (25.6% of
  training-period origin-hour slots contain a later-cancelled flight; 51–58% of flights were
  cancelled on 2013-02-08/09). Announcement times are not in the data. Counts reflect the originally
  scheduled flights and never reveal a flight's own outcome.
- **Real-time outcome availability is assumed:** history windows require flight outcomes within
  about 30 min of arrival (an operational flight-status feed). Public BTS files are released
  monthly with a lag and could not support 1/7/28-day windows. The 30-min buffer is an
  assumption.
- **Time zones are external reference data** (`airportsdata` 20250909, vendored), validated
  against the training period (no negative taxi times) but not part of the dataset.
