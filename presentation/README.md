# Phase 1 Presentation

PPT link: https://drive.google.com/drive/folders/1_z1m0x0Z3fkjQ0yms1E1UcIztOHx0IaW?usp=sharing

TODO: slides (link or file) + speaker notes.

Suggested flow (~10 min):
1. Problem & decision time
2. Data quality & EDA highlights
3. Leakage-safe feature engineering
4. Models: Logistic Regression vs. XGBoost (theory in one slide each)
5. Results: PR AUC, ROC AUC, recall@precision, calibration
6. Limitations & Phase 2 plan

## Final comparison slide: GRU vs LR on November–December (D-313)

Not yet in the slides. `FlightDelayPrediction.pptx (1).pdf` is an export and the editable deck is
on Google Drive, so this slide still has to be added there.

**Title:** Sealed test (Nov–Dec 2013, 53,991 flights, 25.8% delayed): Phase 2 GRU vs Phase 1 LR

| | GRU (Phase 2, D-312) | LR (Phase 1, D-205) |
|---|---|---|
| PR-AUC | 0.399 | 0.398 |
| ROC-AUC | 0.664 | 0.664 |
| Brier (calibrated) | 0.189 | 0.185 |
| ECE-10 (calibrated) | 0.094 | 0.077 |
| At frozen threshold | 1,102 flagged (2.0%), precision 0.574, recall 0.045 | no feasible threshold |

Figure: `report/figures/phase2_gru_test_reliability.png`.

Speaker notes: both models were frozen before the test and scored once. They rank flights the
same, and no interval was pre-registered, so neither is called better. The September calibration
under-predicts in the busier test months for both models. The GRU's 60% precision threshold from
October gave 57.4% on the test period. The deep model matches but does not beat the baseline.

## Viva prep
- [ ] Why PR AUC over accuracy?
- [ ] How do you guarantee no post-departure leakage?
- [ ] Why a chronological split?
- [ ] How does XGBoost differ from a random forest?
- [ ] What does calibration fix and how was it measured?
