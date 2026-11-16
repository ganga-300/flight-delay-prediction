# Phase 2 Report

Plain-language IEEE report on Phase 2 (neural networks vs the Phase 1 baseline). Compile `main.tex`
with pdfLaTeX, for example on Overleaf, by uploading this whole folder.

- Figures in `figures/` are drawn by `scripts/make_phase2_report_figures.py` from the saved results
  in `reports/results/phase2_*` (nothing is retrained):

  ```bash
  python scripts/make_phase2_report_figures.py
  ```

- Every number comes from `docs/decision_log.md` (D-302 to D-313) and the same results files as
  the Phase 2 deck (`presentation/Phase2_FlightDelayPrediction.pptx`), so the report and the deck
  agree.
- Plain-language names used in the text: *ranking score* = average precision (PR-AUC);
  *calibration gap* = expected calibration error (ECE); *real difference* = the 95% range from
  1,000 day-level resamples excludes zero.
