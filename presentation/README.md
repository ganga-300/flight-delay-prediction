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

## Phase 2 deck

`Phase2_FlightDelayPrediction.pptx`: 14 slides in the Phase 1 deck's style (no blue), with speaker
notes on every slide. Flow: title, Phase 2 protocol, two slides per member (M1 simple MLP, M2 GRU,
M3 embedding MLP, M4 ablations and hybrid, M5 common evaluation and the single Nov–Dec test),
conclusions, references. Every number comes from `docs/decision_log.md` (D-301 to D-313) and
`reports/results/phase2_*`. It includes the GRU-vs-LR test comparison (D-313), which replaces
`final_comparison_slide.md` / `final_comparison_slide.pptx`.

## Viva prep
- [ ] Why PR AUC over accuracy?
- [ ] How do you guarantee no post-departure leakage?
- [ ] Why a chronological split?
- [ ] How does XGBoost differ from a random forest?
- [ ] What does calibration fix and how was it measured?
