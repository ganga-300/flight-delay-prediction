"""Train, calibrate, select an operating threshold, and save a model."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay.config import load_config
from flight_delay.data.schema import TARGET_COLUMN
from flight_delay.evaluation.metrics import recall_at_precision
from flight_delay.features.leakage import assert_no_leakage
from flight_delay.models.calibration import calibrate
from flight_delay.models.logistic import build_logistic_pipeline
from flight_delay.models.persistence import save_model
from flight_delay.models.xgboost_model import build_xgboost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    parser.add_argument("--model", choices=["logreg", "xgboost"], required=True)
    parser.add_argument(
        "--calibration",
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = ROOT / config["paths"]["processed"]
    model_dir = ROOT / config["paths"]["models"]
    model_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(data_dir / "train.parquet")
    valid = pd.read_parquet(data_dir / "valid.parquet")
    spec = json.loads((data_dir / "feature_spec.json").read_text())

    target = spec.get("target", TARGET_COLUMN)
    numeric = spec["numeric"]
    categorical = spec["categorical"]
    features = numeric + categorical
    assert_no_leakage(features)

    required = set(features + [target, "date"])
    for name, frame in [("train", train), ("valid", valid)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} split is missing columns: {sorted(missing)}")

    train = train.copy()
    valid = valid.copy()
    train["date"] = pd.to_datetime(train["date"])
    valid["date"] = pd.to_datetime(valid["date"])

    # Explicitly enforce the time-based protocol.
    initial_train = train[train["date"] < "2013-08-01"]
    early_stop = train[
        (train["date"] >= "2013-08-01")
        & (train["date"] < "2013-09-01")
    ]
    final_train = train[train["date"] < "2013-09-01"]
    calibration_data = valid[
        (valid["date"] >= "2013-09-01")
        & (valid["date"] < "2013-10-01")
    ]
    threshold_data = valid[
        (valid["date"] >= "2013-10-01")
        & (valid["date"] < "2013-11-01")
    ]

    for name, frame in [
        ("initial training", initial_train),
        ("August early stopping", early_stop),
        ("final training", final_train),
        ("September calibration", calibration_data),
        ("October threshold selection", threshold_data),
    ]:
        if frame.empty:
            raise ValueError(f"{name} split is empty.")

        if frame[target].nunique() != 2:
            raise ValueError(f"{name} split must contain both target classes.")

    X_initial = initial_train[features]
    y_initial = initial_train[target].astype(int)
    X_early = early_stop[features]
    y_early = early_stop[target].astype(int)
    X_final = final_train[features]
    y_final = final_train[target].astype(int)
    X_cal = calibration_data[features]
    y_cal = calibration_data[target].astype(int)
    X_threshold = threshold_data[features]
    y_threshold = threshold_data[target].astype(int)

    params = dict(config.get("models", {}).get(args.model, {}))

    if args.model == "logreg":
        model = build_logistic_pipeline(numeric, categorical, params)
        model.fit(X_final, y_final)
        selected_trees = None

    else:
        positive = int(y_initial.sum())
        negative = len(y_initial) - positive

        # Find the tree count using August only.
        early_model = build_xgboost(
            params,
            scale_pos_weight=negative / max(positive, 1),
        )
        early_model.fit(
            X_initial,
            y_initial,
            eval_set=[(X_early, y_early)],
            verbose=False,
        )

        selected_trees = int(early_model.best_iteration) + 1
        print(f"XGBoost selected tree count: {selected_trees}")

        # Refit on all January-August data using the selected tree count.
        final_params = dict(params)
        final_params["n_estimators"] = selected_trees
        final_params["early_stopping_rounds"] = None

        positive = int(y_final.sum())
        negative = len(y_final) - positive
        model = build_xgboost(
            final_params,
            scale_pos_weight=negative / max(positive, 1),
        )
        model.fit(X_final, y_final, verbose=False)

    # September is used only to fit the probability calibrator.
    raw_cal_prob = model.predict_proba(X_cal)[:, 1]
    raw_cal_pr_auc = float(average_precision_score(y_cal, raw_cal_prob))

    calibrated_model = calibrate(
        model,
        X_cal,
        y_cal,
        method=args.calibration,
    )

    # October is used only to select the operating threshold.
    threshold_prob = calibrated_model.predict_proba(X_threshold)[:, 1]
    min_precision = float(
        config.get("evaluation", {}).get("operational_precision", 0.60)
    )
    threshold_recall, threshold = recall_at_precision(
        y_threshold.to_numpy(),
        threshold_prob,
        min_precision,
    )

    threshold_is_valid = bool(np.isfinite(threshold))
    saved_threshold = float(threshold) if threshold_is_valid else None

    artifact_path = model_dir / f"{args.model}.joblib"
    save_model(calibrated_model, str(artifact_path))

    metadata = {
        "model": args.model,
        "calibration": args.calibration,
        "features": features,
        "numeric": numeric,
        "categorical": categorical,
        "target": target,
        "validation_pr_auc_before_calibration": raw_cal_pr_auc,
        "calibration_period": "2013-09",
        "threshold_selection_period": "2013-10",
        "early_stopping_period": "2013-08" if args.model == "xgboost" else None,
        "selected_trees": selected_trees,
        "validation_recall_at_precision": float(threshold_recall),
        "operational_precision": min_precision,
        "validation_threshold": saved_threshold,
        "rows_initial_train": len(initial_train),
        "rows_early_stopping": len(early_stop),
        "rows_train": len(final_train),
        "rows_calibration": len(calibration_data),
        "rows_threshold_selection": len(threshold_data),
        "threshold_status": (
            "selected_on_october"
            if threshold_is_valid
            else "no_threshold_met_target_precision"
        ),
    }

    metadata_path = model_dir / f"{args.model}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Model: {args.model}")
    print(f"Initial training rows (Jan-Jul): {len(initial_train)}")
    print(f"Early-stopping rows (August): {len(early_stop)}")
    print(f"Final training rows (Jan-Aug): {len(final_train)}")
    print(f"Calibration rows (September): {len(calibration_data)}")
    print(f"Threshold-selection rows (October): {len(threshold_data)}")
    print(f"September PR-AUC before calibration: {raw_cal_pr_auc:.5f}")
    print(
        f"October recall at precision >= {min_precision:.2f}: "
        f"{threshold_recall:.5f}"
    )
    print(f"October threshold: {saved_threshold}")
    print(f"Saved model: {artifact_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
