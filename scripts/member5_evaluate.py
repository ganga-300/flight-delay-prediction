"""Member 5, PR 1: evaluate the saved September-October predictions under D-203.

Input: ``reports/results/phase2_member4/sep_oct_predictions.parquet`` only (Logistic Regression,
Member 1 MLP, Member 2 GRU, Member 3 D-307 embedding MLP, Member 4 ablations, plain XGBoost and
hybrid; seeds 42-46; 55,628 flights matched by id). Nothing is retrained.

* September: sigmoid calibration fitted per run.
* October:   calibration checked out of sample; threshold chosen at precision >= 0.60
             (infeasible if no threshold reaches it); reliability plots and error analysis.
* September-October: PR-AUC, ROC-AUC and day-level bootstrap differences
             (``scripts/train_gru.day_bootstrap``, 1,000 whole-day resamples, seed 42).

November-December is not read. Carrier, origin and departure hour come from the January-October
development file (``load_dev_raw``), matched on flight id.

Outputs: ``outputs/`` (main models, differences vs Logistic Regression) and ``outputs/ablation/``
(ablations, differences vs the D-307 embedding MLP).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_gru import day_bootstrap  # noqa: E402

from flight_delay.config import load_config  # noqa: E402
from flight_delay.data.load import load_dev_raw  # noqa: E402
from flight_delay.evaluation.evaluate import (  # noqa: E402
    MIN_PRECISION,
    OCTOBER,
    SEPTEMBER,
    attach_meta,
    load_sep_oct,
    run_evaluation,
)

PREDICTIONS = ROOT / "reports/results/phase2_member4/sep_oct_predictions.parquet"
EXPECTED = {"all": 55_628, "september": 27_010, "october": 28_618}

MAIN_MODELS = {
    "lr": "Logistic Regression (Phase 1)",
    "mlp": "Member 1 MLP",
    "gru": "Member 2 GRU",
    "embedding": "Member 3 embedding MLP (D-307)",
    "plain_xgboost": "Plain XGBoost (Member 4)",
    "hybrid_dl_xgboost": "Hybrid DL + XGBoost (Member 4)",
}
ABLATION_MODELS = {
    "embedding": "Member 3 embedding MLP (D-307, reference)",
    "ablation_no_history": "Ablation: no historical features",
    "ablation_no_congestion": "Ablation: no congestion features",
    "ablation_one_hot": "Ablation: one-hot instead of embeddings",
    "ablation_no_regularization": "Ablation: no dropout or weight decay",
}

PROTOCOL = f"""\
Source: `reports/results/phase2_member4/sep_oct_predictions.parquet` (saved probabilities of
models trained on January-August 2013; nothing retrained). Months, each with one job (D-203):

| Period | Flights | Use |
|---|---|---|
| September 2013 | {EXPECTED['september']:,} | Fit one sigmoid calibrator per run |
| October 2013 | {EXPECTED['october']:,} | Out-of-sample calibration check; threshold at precision ≥ {MIN_PRECISION:.2f}; reliability plots; error analysis |
| September-October 2013 | {EXPECTED['all']:,} | PR-AUC, ROC-AUC, bootstrap differences |
| November-December 2013 | not read | Sealed (D-002) |

Each model is reported as mean ± sd over seeds 42-46 (Logistic Regression is one deterministic
run, sd 0).
"""

LIMITATIONS = """\
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
"""

ABLATION_LIMITATIONS = """\
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
"""

ABLATION_INTRO = """\
Member 4's ablations (D-309/D-310) against the D-307 embedding MLP they were designed around.
Differences are each ablation minus the D-307 reference.
"""


def load_inputs():
    cfg = load_config(ROOT / "configs/config.yaml")
    frame = load_sep_oct(PREDICTIONS)
    counts = {
        "all": len(frame),
        "september": int(frame["date"].between(*SEPTEMBER).sum()),
        "october": int(frame["date"].between(*OCTOBER).sum()),
    }
    if counts != EXPECTED:
        raise SystemExit(f"flight counts {counts} differ from {EXPECTED}")
    raw = load_dev_raw(cfg, ROOT)  # January-October only; raises on any later row
    meta = raw[["id", "carrier", "origin", "hour"]].rename(columns={"hour": "dep_hour"})
    return attach_meta(frame, meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", default=str(ROOT / "outputs"))
    parser.add_argument("--only", choices=["all", "main", "ablation"], default="all",
                        help="run only one of the two evaluations")
    args = parser.parse_args()
    out = Path(args.outputs)
    frame = load_inputs()

    if args.only in ("all", "main"):
        main_result = run_evaluation(
            frame, MAIN_MODELS, out, reference="lr", bootstrap_fn=day_bootstrap,
            error_analysis_models=list(MAIN_MODELS),
            title="Member 5 evaluation: September-October 2013 (D-203)",
            intro=PROTOCOL, notes=LIMITATIONS,
        )
        print(main_result.summary[["label", "sepoct_pr_auc_mean", "sepoct_pr_auc_sd",
                                   "oct_sigmoid_brier_mean", "oct_sigmoid_ece_10_mean",
                                   "oct_threshold"]].to_string(index=False))
        print(main_result.bootstrap.to_string(index=False))

    if args.only in ("all", "ablation"):
        ablation_result = run_evaluation(
            frame, ABLATION_MODELS, out / "ablation", reference="embedding",
            bootstrap_fn=day_bootstrap,
            title="Member 5 ablation evaluation: September-October 2013 (D-203)",
            intro=ABLATION_INTRO + "\n" + PROTOCOL, notes=ABLATION_LIMITATIONS,
        )
        print(ablation_result.bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
