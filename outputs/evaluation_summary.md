# Member 5 evaluation: September-October 2013 (D-203)

Source: `reports/results/phase2_member4/sep_oct_predictions.parquet` (saved probabilities of
models trained on January-August 2013; nothing retrained). Months, each with one job (D-203):

| Period | Flights | Use |
|---|---|---|
| September 2013 | 27,010 | Fit one sigmoid calibrator per run |
| October 2013 | 28,618 | Out-of-sample calibration check; threshold at precision ≥ 0.60; reliability plots; error analysis |
| September-October 2013 | 55,628 | PR-AUC, ROC-AUC, bootstrap differences |
| November-December 2013 | not read | Sealed (D-002) |

Each model is reported as mean ± sd over seeds 42-46 (Logistic Regression is one deterministic
run, sd 0).

## Model comparison

Mean ± sd over each model's runs (seeds). Month of each number in the column name. PR-AUC and ROC-AUC are computed on raw probabilities; the September sigmoid does not change the ranking. Calibrated = sigmoid fitted on September, scored on October.

| Model | Runs | PR-AUC (Sep-Oct) | ROC-AUC (Sep-Oct) | Brier raw (Oct) | Brier calibrated (Oct) | ECE raw (Oct) | ECE calibrated (Oct) | Threshold, precision ≥ 0.60 (Oct) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression (Phase 1) | 1 | 0.2896 ± 0.0000 | 0.6800 ± 0.0000 | 0.1917 ± 0.0000 | 0.1360 ± 0.0000 | 0.2270 ± 0.0000 | 0.0274 ± 0.0000 | infeasible |
| Member 1 MLP | 5 | 0.2839 ± 0.0043 | 0.6719 ± 0.0027 | 0.1410 ± 0.0028 | 0.1359 ± 0.0004 | 0.0730 ± 0.0159 | 0.0242 ± 0.0027 | 5/5 feasible: t = 0.416 ± 0.045, flags 1-320 flights, recall 0.0139 |
| Member 2 GRU | 5 | 0.2946 ± 0.0055 | 0.6743 ± 0.0038 | 0.1465 ± 0.0055 | 0.1352 ± 0.0007 | 0.0926 ± 0.0168 | 0.0144 ± 0.0014 | 5/5 feasible: t = 0.405 ± 0.011, flags 160-358 flights, recall 0.0326 |
| Member 3 embedding MLP (D-307) | 5 | 0.2757 ± 0.0035 | 0.6748 ± 0.0017 | 0.1372 ± 0.0003 | 0.1370 ± 0.0005 | 0.0390 ± 0.0061 | 0.0301 ± 0.0019 | infeasible (5/5 runs) |
| Plain XGBoost (Member 4) | 5 | 0.2681 ± 0.0055 | 0.6719 ± 0.0027 | 0.1459 ± 0.0058 | 0.1379 ± 0.0003 | 0.0820 ± 0.0183 | 0.0304 ± 0.0049 | 4/5 feasible: t = 0.481 ± 0.037, flags 1-35 flights, recall 0.0018 |
| Hybrid DL + XGBoost (Member 4) | 5 | 0.2643 ± 0.0101 | 0.6658 ± 0.0071 | 0.1575 ± 0.0221 | 0.1380 ± 0.0005 | 0.1113 ± 0.0464 | 0.0329 ± 0.0020 | 3/5 feasible: t = 0.383 ± 0.035, flags 1-39 flights, recall 0.0021 |

September reference (the calibrator's own fitting month, in-sample):

| Model | Brier raw (Sep) | Brier calibrated (Sep, in-sample) | ECE raw (Sep) | ECE calibrated (Sep, in-sample) |
|---|---|---|---|---|
| Logistic Regression (Phase 1) | 0.1790 ± 0.0000 | 0.1129 ± 0.0000 | 0.2402 ± 0.0000 | 0.0095 ± 0.0000 |
| Member 1 MLP | 0.1251 ± 0.0024 | 0.1144 ± 0.0003 | 0.0809 ± 0.0132 | 0.0086 ± 0.0051 |
| Member 2 GRU | 0.1249 ± 0.0038 | 0.1134 ± 0.0004 | 0.0810 ± 0.0119 | 0.0078 ± 0.0023 |
| Member 3 embedding MLP (D-307) | 0.1188 ± 0.0009 | 0.1134 ± 0.0003 | 0.0634 ± 0.0026 | 0.0085 ± 0.0023 |
| Plain XGBoost (Member 4) | 0.1270 ± 0.0048 | 0.1133 ± 0.0004 | 0.1051 ± 0.0069 | 0.0109 ± 0.0038 |
| Hybrid DL + XGBoost (Member 4) | 0.1426 ± 0.0244 | 0.1143 ± 0.0011 | 0.1280 ± 0.0306 | 0.0120 ± 0.0018 |

## PR-AUC differences vs Logistic Regression (Phase 1)

Paired day-level bootstrap (1,000 resamples of whole days, seed 42) on September-October; each model's PR-AUC is the mean over its seeds. A difference counts only if its 95% interval excludes 0.

| Comparison | PR-AUC difference (Sep-Oct) | 95% interval (Sep-Oct) | Interval excludes 0 |
|---|---|---|---|
| mlp - lr | -0.0057 | [-0.0239, +0.0115] | no |
| gru - lr | +0.0050 | [-0.0137, +0.0238] | no |
| embedding - lr | -0.0139 | [-0.0218, -0.0034] | yes |
| plain_xgboost - lr | -0.0215 | [-0.0385, -0.0038] | yes |
| hybrid_dl_xgboost - lr | -0.0253 | [-0.0395, -0.0105] | yes |

Differences whose interval excludes 0 (September-October): higher: none; lower: embedding - lr, plain_xgboost - lr, hybrid_dl_xgboost - lr. All other differences are within bootstrap noise.

## Error analysis (October, calibrated probabilities)

Groups with at least 200 October flights. Values are means over each model's seeds; per-seed sd is in `error_analysis_groups.csv`. Threshold-based rates are not reported because the October thresholds are infeasible or flag very few flights (D-203).

Lowest PR-AUC groups (October):

| Meta column | Group | Flights (Oct) | Logistic Regression (Phase 1) | Member 1 MLP | Member 2 GRU | Member 3 embedding MLP (D-307) | Plain XGBoost (Member 4) | Hybrid DL + XGBoost (Member 4) | Mean over models |
|---|---|---|---|---|---|---|---|---|---|
| carrier | DL | 4090 | 0.191 | 0.201 | 0.200 | 0.194 | 0.190 | 0.193 | 0.195 |
| carrier | US | 1814 | 0.205 | 0.237 | 0.233 | 0.175 | 0.204 | 0.183 | 0.206 |
| carrier | AA | 2699 | 0.215 | 0.222 | 0.217 | 0.217 | 0.205 | 0.213 | 0.215 |
| origin | JFK | 9096 | 0.227 | 0.222 | 0.213 | 0.221 | 0.203 | 0.209 | 0.216 |
| origin | LGA | 9516 | 0.325 | 0.345 | 0.350 | 0.312 | 0.308 | 0.313 | 0.326 |
| origin | EWR | 10006 | 0.383 | 0.388 | 0.391 | 0.361 | 0.332 | 0.349 | 0.367 |
| dep_hour | 6 | 2107 | 0.132 | 0.117 | 0.111 | 0.119 | 0.129 | 0.116 | 0.121 |
| dep_hour | 7 | 1947 | 0.154 | 0.149 | 0.148 | 0.146 | 0.130 | 0.127 | 0.142 |
| dep_hour | 9 | 1617 | 0.160 | 0.178 | 0.173 | 0.168 | 0.150 | 0.150 | 0.163 |

Largest calibration gaps, mean calibrated probability minus observed rate (October):

| Meta column | Group | Logistic Regression (Phase 1) | Member 1 MLP | Member 2 GRU | Member 3 embedding MLP (D-307) | Plain XGBoost (Member 4) | Hybrid DL + XGBoost (Member 4) | Mean over models |
|---|---|---|---|---|---|---|---|---|
| carrier | FL | -0.140 | -0.153 | -0.121 | -0.150 | -0.151 | -0.170 | -0.147 |
| carrier | EV | -0.074 | -0.067 | -0.050 | -0.074 | -0.068 | -0.083 | -0.069 |
| carrier | WN | -0.062 | -0.058 | -0.048 | -0.074 | -0.078 | -0.080 | -0.067 |
| origin | EWR | -0.048 | -0.035 | -0.027 | -0.040 | -0.038 | -0.041 | -0.038 |
| origin | LGA | -0.016 | -0.020 | +0.000 | -0.033 | -0.034 | -0.033 | -0.023 |
| origin | JFK | -0.016 | -0.016 | -0.012 | -0.015 | -0.017 | -0.021 | -0.016 |
| dep_hour | 19 | -0.061 | -0.067 | -0.052 | -0.066 | -0.068 | -0.078 | -0.065 |
| dep_hour | 15 | -0.064 | -0.066 | -0.052 | -0.068 | -0.064 | -0.076 | -0.065 |
| dep_hour | 17 | -0.049 | -0.051 | -0.037 | -0.049 | -0.043 | -0.056 | -0.047 |

## Limitations

- **Logistic Regression's September-October numbers are not a clean test.** Phase 1 used
  September-October as validation (D-203/D-204: calibration on September, threshold on October),
  so LR's September-October scores are optimistic relative to the Phase 2 models. The fair ML
  baseline trained under the Phase 2 protocol is Member 4's plain XGBoost.
- **October threshold metrics are in-sample.** October chooses the threshold, so the precision,
  recall and flight counts at that threshold are optimistic (D-203).
- **Degenerate feasible thresholds.** D-203 sets no minimum number of flagged flights, so a
  "feasible" October threshold can flag as few as one flight. Flight counts are shown with each
  threshold so this is visible; no minimum has been added.
- **Calibration month differs from the threshold month.** September is the calmest month (13.7%
  delayed vs 17.6% in October), so September-fitted probabilities run low on October.
- **Raw calibration is not comparable across models.** Logistic Regression uses
  `class_weight="balanced"`, which pushes its raw probabilities up (raw October ECE about 0.23).
  Compare calibration only after the September sigmoid.
