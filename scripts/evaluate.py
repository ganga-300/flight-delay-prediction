"""Evaluate calibrated models on the held-out test split."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay.config import load_config
from flight_delay.evaluation.metrics import (
    brier,
    expected_calibration_error,
    pr_auc,
    roc_auc,
)
from flight_delay.models.persistence import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/config.yaml"),
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "xgboost", "all"],
        default="all",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    data_dir = ROOT / config["paths"]["processed"]
    model_dir = ROOT / config["paths"]["models"]
    results_dir = ROOT / config["paths"]["results"]
    figures_dir = ROOT / config["paths"]["figures"]

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    test = pd.read_parquet(data_dir / "test.parquet")
    spec = json.loads(
        (data_dir / "feature_spec.json").read_text()
    )

    target = spec.get("target", "is_delayed")
    features = spec["numeric"] + spec["categorical"]

    X_test = test[features]
    y_test = test[target].astype(int)

    min_precision = float(
        config.get("evaluation", {}).get(
            "operational_precision", 0.60
        )
    )
    n_bins = int(
        config.get("evaluation", {}).get(
            "calibration_bins", 10
        )
    )

    names = (
        ["logreg", "xgboost"]
        if args.model == "all"
        else [args.model]
    )

    rows = []

    for name in names:
        model_path = model_dir / f"{name}.joblib"
        metadata_path = model_dir / f"{name}_metadata.json"

        if not model_path.is_file() or not metadata_path.is_file():
            print(
                f"Skipping {name}: model or metadata not found. "
                "Train it first."
            )
            continue

        metadata = json.loads(metadata_path.read_text())
        model = load_model(str(model_path))

        probability = model.predict_proba(X_test)[:, 1]

        # Apply the threshold selected using validation data.
        threshold = metadata.get("validation_threshold")

        if threshold is None:
            test_recall = None
            applied_threshold = None
            threshold_status = (
                "No threshold met the target precision on validation"
            )
        else:
            applied_threshold = float(threshold)

            predictions = (
                probability >= applied_threshold
            ).astype(int)

            y_test_array = y_test.to_numpy()

            tp = int(
                (
                    (predictions == 1)
                    & (y_test_array == 1)
                ).sum()
            )
            fn = int(
                (
                    (predictions == 0)
                    & (y_test_array == 1)
                ).sum()
            )

            test_recall = (
                tp / (tp + fn)
                if tp + fn
                else 0.0
            )

            threshold_status = (
                "Target precision threshold selected on validation"
            )

        # Calculate evaluation metrics.
        result = {
            "model": name,
            "calibration": metadata["calibration"],
            "test_pr_auc": pr_auc(y_test, probability),
            "test_roc_auc": roc_auc(y_test, probability),
            "test_recall_at_validation_threshold": test_recall,
            "validation_threshold": applied_threshold,
            "target_precision": min_precision,
            "threshold_status": threshold_status,
            "test_brier": brier(y_test, probability),
            "test_ece": expected_calibration_error(
                y_test,
                probability,
                n_bins=n_bins,
            ),
            "rows_test": len(test),
            "positive_rate_test": float(y_test.mean()),
        }

        rows.append(result)

        # Generate the reliability diagram.
        fraction_positive, mean_predicted = calibration_curve(
            y_test,
            probability,
            n_bins=n_bins,
            strategy="uniform",
        )

        plt.figure()
        plt.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            label=name,
        )
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Perfect calibration",
        )
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed positive fraction")
        plt.title(f"Reliability diagram — {name}")
        plt.legend()
        plt.tight_layout()

        figure_path = figures_dir / f"{name}_reliability.png"
        plt.savefig(figure_path, dpi=150)
        plt.close()

        print(f"Saved reliability diagram: {figure_path}")
        print(json.dumps(result, indent=2))

    if not rows:
        raise FileNotFoundError(
            f"No trained models found in {model_dir}. "
            "Run scripts/train.py first."
        )

    # Save the evaluation results.
    output = results_dir / "test_metrics.json"
    output.write_text(json.dumps(rows, indent=2))

    csv_output = results_dir / "test_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_output, index=False)

    print(f"Saved test metrics: {output}")
    print(f"Saved test metrics CSV: {csv_output}")


if __name__ == "__main__":
    main()
