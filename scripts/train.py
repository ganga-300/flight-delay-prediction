"""Train Logistic Regression or XGBoost, calibrate on validation, and save artifacts."""

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
from flight_delay.features.leakage import assert_no_leakage
from flight_delay.data.schema import TARGET_COLUMN
from flight_delay.evaluation.metrics import recall_at_precision
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

    for name, frame in [("train", train), ("valid", valid)]:
        missing = set(features + [target]) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} split is missing columns: {sorted(missing)}")

    X_train, y_train = train[features], train[target].astype(int)
    X_valid, y_valid = valid[features], valid[target].astype(int)

    if y_train.nunique() < 2:
        raise ValueError("Training data must contain both target classes.")
    if y_valid.nunique() < 2:
        raise ValueError("Validation data must contain both target classes.")

    params = config.get("models", {}).get(args.model, {})

    if args.model == "logreg":
        model = build_logistic_pipeline(numeric, categorical, params)
        model.fit(X_train, y_train)
    else:
        positive = int(y_train.sum())
        negative = len(y_train) - positive
        model = build_xgboost(
            params,
            scale_pos_weight=negative / max(positive, 1),
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

    raw_valid_prob = model.predict_proba(X_valid)[:, 1]
    raw_pr_auc = float(average_precision_score(y_valid, raw_valid_prob))

    calibrated_model = calibrate(
        model,
        X_valid,
        y_valid,
        method=args.calibration,
    )
    calibrated_valid_prob = calibrated_model.predict_proba(X_valid)[:, 1]

    min_precision = float(
        config.get("evaluation", {}).get("operational_precision", 0.60)
    )
    valid_recall, threshold = recall_at_precision(
        y_valid,
        calibrated_valid_prob,
        min_precision,
    )

    artifact_path = model_dir / f"{args.model}.joblib"
    save_model(calibrated_model, str(artifact_path))

    metadata = {
        "model": args.model,
        "calibration": args.calibration,
        "features": features,
        "numeric": numeric,
        "categorical": categorical,
        "target": target,
        "validation_pr_auc_before_calibration": raw_pr_auc,
        "validation_recall_at_precision": valid_recall,
        "operational_precision": min_precision,
        "validation_threshold": threshold if np.isfinite(threshold) else None,
        "rows_train": len(train),
        "rows_valid": len(valid),
    }
    metadata_path = model_dir / f"{args.model}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Model: {args.model}")
    print(f"Validation PR-AUC before calibration: {raw_pr_auc:.5f}")
    print(f"Validation recall at precision >= {min_precision:.2f}: {valid_recall:.5f}")
    print(f"Saved model: {artifact_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
