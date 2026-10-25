"""D-203 Part D: calibrate on September, choose the threshold and apply the rescheduling rule on
October. Writes the frozen calibrated model and operating point for the single test evaluation.

Uses only the September and October rows of data/processed/valid.parquet. The final D-202 model
is used unchanged. Refuses to overwrite earlier outputs, so the result cannot be quietly redone.
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay.config import load_config  # noqa: E402
from flight_delay.evaluation.metrics import (  # noqa: E402
    brier,
    expected_calibration_error,
    pr_auc,
    roc_auc,
)
from flight_delay.models.calibration import calibrate  # noqa: E402
from flight_delay.models.operating_point import choose_threshold  # noqa: E402
from flight_delay.models.persistence import load_model, save_model  # noqa: E402

PLAN_COMMIT = "968067b"
SEPTEMBER = ("2013-09-01", "2013-09-30")
OCTOBER = ("2013-10-01", "2013-10-31")
MODEL = ROOT / "models/d201/final_model.joblib"
MODEL_ENCODER = ROOT / "models/d201/final_encoder.json"
OUT_MODELS = ROOT / "models/d203"
OUT_RESULTS = ROOT / "reports/results"
OUT_FIGURES = ROOT / "reports/figures"
TARGET = "is_delayed"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(y, p) -> dict:
    return {
        "pr_auc": pr_auc(y, p),
        "roc_auc": roc_auc(y, p),
        "brier": brier(y, p),
        "ece_10": expected_calibration_error(y, p, n_bins=10),
        "mean_probability": float(np.mean(p)),
        "delay_rate": float(np.mean(y)),
    }


def reliability_plot(y, probs: dict, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    for name, p in probs.items():
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="uniform")
        ax.plot(predicted, observed, marker="o", label=name)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed delay rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    processed = ROOT / cfg["paths"]["processed"]
    outputs = [OUT_MODELS / "calibrated_model.joblib", OUT_MODELS / "operating_point.json"]
    if any(p.exists() for p in outputs):
        raise SystemExit("D-203 outputs already exist; refusing to overwrite")

    # The processed validation data must be encoded exactly as the final model expects.
    model_encoder = json.loads(MODEL_ENCODER.read_text())
    if model_encoder != json.loads((processed / "encoder.json").read_text()):
        raise SystemExit("valid.parquet encoder differs from the final model's encoder")

    spec = json.loads((processed / "feature_spec.json").read_text())
    features = list(spec["numeric"]) + list(spec["categorical"])
    if len(features) != 59 or len(set(features)) != 59:
        raise SystemExit("expected the 59 frozen features")

    valid = pd.read_parquet(processed / "valid.parquet")
    dates = pd.to_datetime(valid["date"])
    if not dates.between(SEPTEMBER[0], OCTOBER[1]).all():
        raise SystemExit("valid.parquet holds rows outside September-October")
    sep = valid[dates.between(*SEPTEMBER)]
    octo = valid[dates.between(*OCTOBER)]
    X_sep, y_sep = sep[features], sep[TARGET].astype(int).to_numpy()
    X_oct, y_oct = octo[features], octo[TARGET].astype(int).to_numpy()

    model = load_model(MODEL)
    sigmoid = calibrate(model, X_sep, y_sep, method="sigmoid")
    isotonic = calibrate(model, X_sep, y_sep, method="isotonic")  # comparison only

    probs = {
        period: {
            "raw": model.predict_proba(X)[:, 1],
            "sigmoid": sigmoid.predict_proba(X)[:, 1],
            "isotonic (comparison)": isotonic.predict_proba(X)[:, 1],
        }
        for period, X in (("september", X_sep), ("october", X_oct))
    }
    truth = {"september": y_sep, "october": y_oct}
    rows = [
        {"period": period, "probabilities": name, **scores(truth[period], p)}
        for period, by_name in probs.items()
        for name, p in by_name.items()
    ]

    min_precision = float(cfg["evaluation"]["operational_precision"])
    p_oct = probs["october"]["sigmoid"]
    op = choose_threshold(y_oct, p_oct, min_precision=min_precision)
    if op["feasible"]:
        flagged = p_oct >= op["threshold"]
        rescheduling = {
            "flights": int(len(p_oct)),
            "flagged": int(flagged.sum()),
            "share_flagged": float(flagged.mean()),
            "flagged_actually_delayed": float(y_oct[flagged].mean()),
            "delayed_flights_flagged": float(flagged[y_oct == 1].mean()),
        }
    else:
        rescheduling = {"flights": int(len(p_oct)), "status": "not applied: no feasible threshold"}

    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    save_model(sigmoid, OUT_MODELS / "calibrated_model.joblib")
    record = {
        "protocol": "D-203",
        "plan_commit": PLAN_COMMIT,
        "base_model": "models/d201/final_model.joblib",
        "base_model_sha256": sha256(MODEL),
        "calibration": {"method": "sigmoid", "period": SEPTEMBER, "rows": int(len(y_sep))},
        "threshold_period": OCTOBER,
        "min_precision": min_precision,
        "operating_point": op,
        "rescheduling_rule": "flag flights with calibrated probability >= threshold for a buffer",
        "rescheduling_october": rescheduling,
        "features": features,
    }
    (OUT_MODELS / "operating_point.json").write_text(json.dumps(record, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(OUT_RESULTS / "d203_calibration.csv", index=False)
    for period in ("september", "october"):
        reliability_plot(
            truth[period],
            probs[period],
            f"Reliability — {period.capitalize()} 2013",
            OUT_FIGURES / f"d203_reliability_{period}.png",
        )

    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("operating point:", json.dumps(op))
    print("rescheduling (October):", json.dumps(rescheduling))


if __name__ == "__main__":
    main()
