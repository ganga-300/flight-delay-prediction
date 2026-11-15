# Final comparison slide (ready to paste into the Drive deck)

Source: D-313 (`docs/decision_log.md`) and
`reports/results/phase2_member5/test_nov_dec/test_metrics.json`.
Figure (optional): `report/figures/phase2_gru_test_reliability.png`.

---

## Slide title

Sealed test (Nov–Dec 2013, 53,991 flights, 25.8% delayed): Phase 2 GRU vs Phase 1 LR

## Table

| | GRU (Phase 2, D-312) | LR (Phase 1, D-205) |
|---|---|---|
| PR-AUC | 0.3986 | 0.3979 |
| ROC-AUC | 0.6638 | 0.6640 |
| Brier (raw / calibrated) | 0.1869 / 0.1890 | 0.2264 / 0.1852 |
| ECE-10 (raw / calibrated) | 0.0592 / 0.0941 | 0.2039 / 0.0768 |
| At frozen threshold (0.394665) | 1,102 flagged (2.04%), precision 0.5735, recall 0.0453 | not available (no feasible threshold on October, D-204) |

## Bullets

- Both models were frozen before the test and scored once on the same sealed flights.
- Ranking is almost identical (PR-AUC 0.3986 vs 0.3979). No interval was pre-registered, so neither model is called better.
- The September calibration under-predicts in the busier test months: the GRU's calibrated ECE-10 rose from 0.0592 (raw) to 0.0941, and its mean calibrated probability was 0.164 against a 0.258 delay rate.
- The GRU's October threshold (60% precision in-sample) gave 57.4% precision on the test period, below the target.

## Speaker notes

Both models were frozen before the test and scored once. They rank flights the same, and no
interval was pre-registered, so neither is called better. The September calibration
under-predicts in the busier test months for both models. The GRU's 60% precision threshold from
October gave 57.4% on the test period. The deep model matches but does not beat the baseline.
