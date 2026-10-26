# IEEE Project Report

Built with the `IEEEtran` conference class. One file per section in `sections/`. The compiled
report is `FlightDelayPrediction.pdf` (compiled on Overleaf with pdfLaTeX).

```bash
cd report
latexmk -pdf main.tex     # or upload the folder to overleaf
```

Figures are looked up in `report/figures/` first, then `reports/figures/`. The two figures used in
the report are copied into `report/figures/` so the folder compiles on its own (e.g. on Overleaf):
`d201_pr_comparison.png` (from `scripts/report_d201.py`) and `test_reliability.png` (from
`scripts/evaluate.py`). All numbers in the report come from `docs/decision_log.md` and
`reports/results/`.
