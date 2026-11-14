# Member 5 ablation evaluation: September-October 2013 (D-203)

Member 4's ablations (D-309/D-310) against the D-307 embedding MLP they were designed around.
Differences are each ablation minus the D-307 reference.

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
| Member 3 embedding MLP (D-307, reference) | 5 | 0.2757 ± 0.0035 | 0.6748 ± 0.0017 | 0.1372 ± 0.0003 | 0.1370 ± 0.0005 | 0.0390 ± 0.0061 | 0.0301 ± 0.0019 | infeasible (5/5 runs) |
| Ablation: no historical features | 5 | 0.2576 ± 0.0062 | 0.6539 ± 0.0043 | 0.1475 ± 0.0009 | 0.1387 ± 0.0008 | 0.0957 ± 0.0028 | 0.0318 ± 0.0020 | 4/5 feasible: t = 0.306 ± 0.017, flags 14-308 flights, recall 0.0128 |
| Ablation: no congestion features | 5 | 0.2596 ± 0.0037 | 0.6675 ± 0.0025 | 0.1379 ± 0.0005 | 0.1384 ± 0.0004 | 0.0333 ± 0.0054 | 0.0350 ± 0.0014 | 1/5 feasible: t = 0.394 ± 0.000, flags 3 flights, recall 0.0004 |
| Ablation: one-hot instead of embeddings | 5 | 0.2753 ± 0.0033 | 0.6770 ± 0.0015 | 0.1370 ± 0.0004 | 0.1370 ± 0.0003 | 0.0377 ± 0.0041 | 0.0317 ± 0.0012 | 4/5 feasible: t = 0.401 ± 0.018, flags 1-11 flights, recall 0.0007 |
| Ablation: no dropout or weight decay | 5 | 0.2757 ± 0.0021 | 0.6735 ± 0.0029 | 0.1384 ± 0.0020 | 0.1370 ± 0.0004 | 0.0474 ± 0.0149 | 0.0286 ± 0.0037 | 1/5 feasible: t = 0.432 ± 0.000, flags 8 flights, recall 0.0010 |

September reference (the calibrator's own fitting month, in-sample):

| Model | Brier raw (Sep) | Brier calibrated (Sep, in-sample) | ECE raw (Sep) | ECE calibrated (Sep, in-sample) |
|---|---|---|---|---|
| Member 3 embedding MLP (D-307, reference) | 0.1188 ± 0.0009 | 0.1134 ± 0.0003 | 0.0634 ± 0.0026 | 0.0085 ± 0.0023 |
| Ablation: no historical features | 0.1341 ± 0.0006 | 0.1154 ± 0.0001 | 0.1185 ± 0.0032 | 0.0130 ± 0.0039 |
| Ablation: no congestion features | 0.1196 ± 0.0004 | 0.1140 ± 0.0002 | 0.0662 ± 0.0040 | 0.0075 ± 0.0020 |
| Ablation: one-hot instead of embeddings | 0.1187 ± 0.0005 | 0.1133 ± 0.0002 | 0.0661 ± 0.0029 | 0.0104 ± 0.0017 |
| Ablation: no dropout or weight decay | 0.1200 ± 0.0025 | 0.1135 ± 0.0004 | 0.0681 ± 0.0059 | 0.0087 ± 0.0015 |

## PR-AUC differences vs Member 3 embedding MLP (D-307, reference)

Paired day-level bootstrap (1,000 resamples of whole days, seed 42) on September-October; each model's PR-AUC is the mean over its seeds. A difference counts only if its 95% interval excludes 0.

| Comparison | PR-AUC difference (Sep-Oct) | 95% interval (Sep-Oct) | Interval excludes 0 |
|---|---|---|---|
| ablation_no_history - embedding | -0.0181 | [-0.0462, +0.0056] | no |
| ablation_no_congestion - embedding | -0.0161 | [-0.0267, -0.0057] | yes |
| ablation_one_hot - embedding | -0.0004 | [-0.0064, +0.0051] | no |
| ablation_no_regularization - embedding | -0.0000 | [-0.0028, +0.0028] | no |

Differences whose interval excludes 0 (September-October): higher: none; lower: ablation_no_congestion - embedding. All other differences are within bootstrap noise.

## Limitations

- **Differences are relative to one reference.** Each ablation is compared with the D-307
  embedding MLP it was designed around (D-309/D-310), not with Logistic Regression or other
  models; see `outputs/evaluation_summary.md` for the main comparison.
- **October threshold metrics are in-sample.** October chooses the threshold, so the precision,
  recall and flight counts at that threshold are optimistic (D-203).
- **Degenerate feasible thresholds.** D-203 sets no minimum number of flagged flights, so a
  "feasible" October threshold can flag as few as one flight. Flight counts are shown with each
  threshold so this is visible; no minimum has been added.
- **Calibration month differs from the threshold month.** September is the calmest month (13.7%
  delayed vs 17.6% in October), so September-fitted probabilities run low on October.
- **No error analysis here.** Reliability plots and group error analysis are produced for the
  main models only (`outputs/`).
