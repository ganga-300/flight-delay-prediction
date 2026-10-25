"""Single test evaluation on the sealed Nov-Dec holdout (docs/decision_log.md, D-203, D-205).

Run once, after D-204 is committed and tagged ``model-frozen``. Before reading any holdout row the
script checks that:

* the ``model-frozen`` tag exists and the frozen files on disk are byte-identical to the tagged
  ones (base model, its encoder, calibrated model, operating point);
* no test results exist yet, so the holdout is evaluated once;
* the full-year file matches the SHA-256 recorded in D-002.

Features are built over the whole year so test flights get their correct past-only history, then
only ``split == "test"`` rows are scored. Nothing is fitted here.
"""

import hashlib
import json
import subprocess
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
from flight_delay.data.load import load_raw  # noqa: E402
from flight_delay.evaluation.metrics import (  # noqa: E402
    brier,
    expected_calibration_error,
    pr_auc,
    roc_auc,
)
from flight_delay.features.build import build_feature_table  # noqa: E402
from flight_delay.features.encoding import FeatureEncoder  # noqa: E402
from flight_delay.models.persistence import load_model  # noqa: E402

SEALED_SHA256 = "8bdc6abbf07ac0cbb1036bc14a787cab3eea616be46c31899d2d8048963b08aa"
FROZEN_TAG = "model-frozen"
FROZEN_FILES = [
    "models/d201/final_model.joblib",
    "models/d201/final_encoder.json",
    "models/d203/calibrated_model.joblib",
    "models/d203/operating_point.json",
]
RESULTS_JSON = ROOT / "reports/results/test_metrics.json"
RESULTS_CSV = ROOT / "reports/results/test_metrics.csv"
FIGURE = ROOT / "reports/figures/test_reliability.png"
TARGET = "is_delayed"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_frozen_state() -> str:
    """Return the frozen commit, or exit if the evaluation must not run."""
    if RESULTS_JSON.exists() or RESULTS_CSV.exists():
        raise SystemExit("Test results already exist: the holdout has been evaluated once.")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", f"{FROZEN_TAG}^{{commit}}"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        raise SystemExit(f"Tag {FROZEN_TAG!r} not found: freeze D-204 first.") from None
    for rel in FROZEN_FILES:
        tagged = subprocess.run(
            ["git", "show", f"{FROZEN_TAG}:{rel}"], cwd=ROOT, capture_output=True, check=True
        ).stdout
        if (ROOT / rel).read_bytes() != tagged:
            raise SystemExit(f"{rel} differs from the {FROZEN_TAG} version")
    return commit


def scores(y, p) -> dict:
    return {
        "flights": int(len(y)),
        "delay_rate": float(np.mean(y)),
        "mean_probability": float(np.mean(p)),
        "pr_auc": pr_auc(y, p),
        "roc_auc": roc_auc(y, p),
        "brier": brier(y, p),
        "ece_10": expected_calibration_error(y, p, n_bins=10),
    }


def main() -> None:
    frozen_commit = check_frozen_state()
    cfg = load_config()
    op = json.loads((ROOT / "models/d203/operating_point.json").read_text())
    features = op["features"]

    full_path = ROOT / cfg["paths"]["raw_sealed"]
    if sha256_bytes(full_path.read_bytes()) != SEALED_SHA256:
        raise SystemExit(f"{full_path} does not match the SHA-256 recorded in D-002")

    table = build_feature_table(load_raw(full_path), cfg)
    test = table[table["split"] == "test"].reset_index(drop=True)
    dates = pd.to_datetime(test["date"])
    start, end = (pd.Timestamp(d) for d in cfg["split"]["test"])
    if test.empty or not dates.between(start, end).all():
        raise SystemExit("unexpected test rows")

    encoder = FeatureEncoder.load(ROOT / "models/d201/final_encoder.json")
    X = encoder.transform(test)[features]
    y = test[TARGET].astype(int).to_numpy()

    base = load_model(ROOT / "models/d201/final_model.joblib")
    calibrated = load_model(ROOT / "models/d203/calibrated_model.joblib")
    probs = {
        "raw": base.predict_proba(X)[:, 1],
        "sigmoid (frozen)": calibrated.predict_proba(X)[:, 1],
    }
    rows = [{"probabilities": name, **scores(y, p)} for name, p in probs.items()]

    point = op["operating_point"]
    if point["feasible"]:
        flagged = probs["sigmoid (frozen)"] >= point["threshold"]
        at_threshold = {
            "threshold": point["threshold"],
            "flagged": int(flagged.sum()),
            "precision": float(y[flagged].mean()) if flagged.any() else None,
            "recall": float(flagged[y == 1].mean()),
        }
        rescheduling = {"flagged_for_buffer": int(flagged.sum())}
    else:
        at_threshold = "not available: no feasible threshold at 0.60 precision (D-204)"
        rescheduling = "not applied: no feasible threshold (D-204)"

    result = {
        "protocol": "D-203",
        "frozen_commit": frozen_commit,
        "test_period": cfg["split"]["test"],
        "metrics": rows,
        "at_threshold": at_threshold,
        "rescheduling_rule": rescheduling,
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    for name, p in probs.items():
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="uniform")
        ax.plot(predicted, observed, marker="o", label=name)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed delay rate")
    ax.set_title("Reliability — test period (Nov–Dec 2013)")
    ax.legend()
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)

    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("at threshold:", at_threshold)
    print("rescheduling:", rescheduling)
    print(f"Saved {RESULTS_JSON.relative_to(ROOT)}, {RESULTS_CSV.relative_to(ROOT)}, "
          f"{FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
