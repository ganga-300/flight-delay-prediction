"""Compare feature groups using temporally separated model-selection stages."""

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
from flight_delay.evaluation.metrics import (
    brier,
    expected_calibration_error,
    pr_auc,
    recall_at_precision,
    roc_auc,
)
from flight_delay.features.leakage import assert_no_leakage
from flight_delay.models.calibration import calibrate
from flight_delay.models.logistic import build_logistic_pipeline
from flight_delay.models.xgboost_model import build_xgboost


EXPERIMENTS = {
    # Existing experiments: preserve for comparison.
    "baseline": ["calendar", "schedule_route", "categorical"],
    "no_schedule_route": ["calendar", "categorical"],
    "no_calendar": ["schedule_route", "categorical"],

    # New experiments: measure the incremental value of feature groups.
    "baseline_plus_historical": [
        "calendar", "schedule_route", "categorical", "historical"
    ],
    "baseline_plus_congestion": [
        "calendar", "schedule_route", "categorical", "congestion"
    ],
    "baseline_plus_historical_congestion": [
        "calendar", "schedule_route", "categorical",
        "historical", "congestion"
    ],
}


def select_features(spec, groups):
    """Select numeric and categorical columns from feature groups."""
    feature_groups = spec["feature_groups"]

    unknown = set(groups) - set(feature_groups)
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")

    allowed = {
        column
        for group in groups
        for column in feature_groups[group]
    }

    numeric = [c for c in spec["numeric"] if c in allowed]
    categorical = [c for c in spec["categorical"] if c in allowed]
    features = numeric + categorical

    if not features:
        raise ValueError(f"No features selected for groups: {groups}")

    if len(features) != len(set(features)):
        raise ValueError("Duplicate features found in selected feature list.")

    assert_no_leakage(features)

    missing_from_spec = allowed - set(features)
    if missing_from_spec:
        raise ValueError(
            "Feature groups contain columns absent from numeric/categorical "
            f"feature lists: {sorted(missing_from_spec)}"
        )

    return numeric, categorical, features


def check_period(frame, target, name):
    """Validate a nonempty period with both target classes."""
    if frame.empty:
        raise ValueError(f"{name} split is empty.")

    if frame[target].nunique() != 2:
        raise ValueError(f"{name} split must contain both target classes.")


def evaluate_variant(
    model_name,
    variant_name,
    groups,
    numeric,
    categorical,
    features,
    train,
    valid,
    test,
    target,
    config,
    calibration_method,
):
    """Train, calibrate and select a threshold on distinct chronological periods."""

    # Expected split files:
    # train.parquet: January-August 2013
    # valid.parquet: September-October 2013
    # test.parquet: November-December 2013
    train = train.copy()
    valid = valid.copy()
    test = test.copy()

    for name, frame in [
        ("train", train),
        ("valid", valid),
        ("test", test),
    ]:
        required = set(features + [target, "date"])
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} split is missing columns: {sorted(missing)}"
            )
        frame["date"] = pd.to_datetime(frame["date"])

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
        ("January-July initial training", initial_train),
        ("August early stopping", early_stop),
        ("January-August final training", final_train),
        ("September calibration", calibration_data),
        ("October threshold selection", threshold_data),
        ("November-December test", test),
    ]:
        check_period(frame, target, name)

    X_initial = initial_train[features]
    y_initial = initial_train[target].astype(int)

    X_final = final_train[features]
    y_final = final_train[target].astype(int)

    X_cal = calibration_data[features]
    y_cal = calibration_data[target].astype(int)

    X_threshold = threshold_data[features]
    y_threshold = threshold_data[target].astype(int)

    X_test = test[features]
    y_test = test[target].astype(int)

    params = dict(config.get("models", {}).get(model_name, {}))
    selected_trees = None

    if model_name == "logreg":
        # Logistic regression does not use early stopping.
        model = build_logistic_pipeline(numeric, categorical, params)
        model.fit(X_final, y_final)

    elif model_name == "xgboost":
        # Select the tree count using August only.
        positive = int(y_initial.sum())
        negative = len(y_initial) - positive

        early_model = build_xgboost(
            params,
            scale_pos_weight=negative / max(positive, 1),
        )
        early_model.fit(
            X_initial,
            y_initial,
            eval_set=[(early_stop[features], early_stop[target].astype(int))],
            verbose=False,
        )

        best_iteration = getattr(early_model, "best_iteration", None)
        if best_iteration is None:
            raise RuntimeError(
                "XGBoost did not expose best_iteration. Check that "
                "early stopping is enabled in the model configuration."
            )

        selected_trees = int(best_iteration) + 1
        print(f"Selected tree count using August: {selected_trees}")

        # Refit on January-August with the selected tree count.
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

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # September is reserved for calibration, not threshold selection.
    raw_cal_probability = model.predict_proba(X_cal)[:, 1]
    september_pr_auc = float(
        average_precision_score(y_cal, raw_cal_probability)
    )

    calibrated_model = calibrate(
        model,
        X_cal,
        y_cal,
        method=calibration_method,
    )

    # October is used only to choose the operational threshold.
    threshold_probability = calibrated_model.predict_proba(X_threshold)[:, 1]

    evaluation_config = config.get("evaluation", {})
    target_precision = float(
        evaluation_config.get("operational_precision", 0.60)
    )
    calibration_bins = int(evaluation_config.get("calibration_bins", 10))

    october_recall, threshold = recall_at_precision(
        y_threshold.to_numpy(),
        threshold_probability,
        target_precision,
    )

    threshold_is_valid = bool(np.isfinite(threshold))

    # November-December remains untouched until final evaluation.
    test_probability = calibrated_model.predict_proba(X_test)[:, 1]

    if threshold_is_valid:
        predictions = (test_probability >= threshold).astype(int)
        actual = y_test.to_numpy()

        tp = int(((predictions == 1) & (actual == 1)).sum())
        fp = int(((predictions == 1) & (actual == 0)).sum())
        tn = int(((predictions == 0) & (actual == 0)).sum())
        fn = int(((predictions == 0) & (actual == 1)).sum())

        test_precision = tp / (tp + fp) if tp + fp else 0.0
        test_recall = tp / (tp + fn) if tp + fn else 0.0
        predicted_positive_count = int(predictions.sum())
        applied_threshold = float(threshold)

        threshold_status = "selected_on_october"
    else:
        tp = fp = tn = fn = None
        test_precision = test_recall = None
        predicted_positive_count = None
        applied_threshold = None

        threshold_status = "no_october_threshold_met_target_precision"

    return {
        "experiment": variant_name,
        "feature_groups": groups,
        "model": model_name,
        "calibration": calibration_method,
        "feature_count": len(features),
        "numeric_feature_count": len(numeric),
        "categorical_feature_count": len(categorical),
        "early_stopping_period": "2013-08" if model_name == "xgboost" else None,
        "selected_trees": selected_trees,
        "calibration_period": "2013-09",
        "threshold_selection_period": "2013-10",
        "test_period": "2013-11_to_2013-12",
        "september_pr_auc_before_calibration": september_pr_auc,
        "october_recall_at_target_precision": float(october_recall),
        "october_threshold": applied_threshold,
        "target_precision": target_precision,
        "threshold_status": threshold_status,
        "test_pr_auc": pr_auc(y_test.to_numpy(), test_probability),
        "test_roc_auc": roc_auc(y_test.to_numpy(), test_probability),
        "test_precision_at_october_threshold": test_precision,
        "test_recall_at_october_threshold": test_recall,
        "test_true_positives": tp,
        "test_false_positives": fp,
        "test_true_negatives": tn,
        "test_false_negatives": fn,
        "test_predicted_positive_count": predicted_positive_count,
        "test_brier": brier(y_test.to_numpy(), test_probability),
        "test_ece": expected_calibration_error(
            y_test.to_numpy(),
            test_probability,
            n_bins=calibration_bins,
        ),
        "rows_initial_train_jan_jul": len(initial_train),
        "rows_early_stop_august": len(early_stop),
        "rows_final_train_jan_aug": len(final_train),
        "rows_calibration_september": len(calibration_data),
        "rows_threshold_selection_october": len(threshold_data),
        "rows_test_nov_dec": len(test),
        "positive_rate_test": float(y_test.mean()),
    }


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
    parser.add_argument(
        "--calibration",
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
    )
    parser.add_argument(
        "--output-name",
        default="ablation_v3",
        help="Output subdirectory under the configured results directory.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = ROOT / config["paths"]["processed"]

    train = pd.read_parquet(data_dir / "train.parquet")
    valid = pd.read_parquet(data_dir / "valid.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")

    spec = json.loads(
        (data_dir / "feature_spec.json").read_text()
    )
    target = spec.get("target", TARGET_COLUMN)

    for name, frame in [
        ("train", train),
        ("valid", valid),
        ("test", test),
    ]:
        if target not in frame.columns:
            raise ValueError(f"{name} split is missing target {targetrm -- "="}.")

    models = (
        ["logreg", "xgboost"]
        if args.model == "all"
        else [args.model]
    )

    results = []

    for variant_name, groups in EXPERIMENTS.items():
        numeric, categorical, features = select_features(spec, groups)

        for model_name in models:
            print(
                f"\nRunning experiment={variant_name}, "
                f"model={model_name}, features={len(features)}",
                flush=True,
            )

            result = evaluate_variant(
                model_name=model_name,
                variant_name=variant_name,
                groups=groups,
                numeric=numeric,
                categorical=categorical,
                features=features,
                train=train,
                valid=valid,
                test=test,
                target=target,
                config=config,
                calibration_method=args.calibration,
            )
            results.append(result)

            print(
                json.dumps(
                    {
                        "experiment": variant_name,
                        "model": model_name,
                        "september_pr_auc_before_calibration": (
                            result["september_pr_auc_before_calibration"]
                        ),
                        "october_recall_at_target_precision": (
                            result["october_recall_at_target_precision"]
                        ),
                        "october_threshold": result["october_threshold"],
                        "test_pr_auc": result["test_pr_auc"],
                        "test_roc_auc": result["test_roc_auc"],
                        "test_brier": result["test_brier"],
                        "test_ece": result["test_ece"],
                        "test_precision": (
                            result["test_precision_at_october_threshold"]
                        ),
                        "test_recall": (
                            result["test_recall_at_october_threshold"]
                        ),
                    },
                    indent=2,
                ),
                flush=True,
            )

    # Save to a named directory so existing experiment results are preserved.
    if Path(args.output_name).name != args.output_name or args.output_name in {".", ".."}:
        raise ValueError("--output-name must be a simple directory name.")

    output_dir = ROOT / config["paths"]["results"] / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"

    json_path.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    pd.DataFrame(results).to_csv(csv_path, index=False)

    print(f"\nSaved ablation results: {json_path}")
    print(f"Saved ablation CSV: {csv_path}")


if __name__ == "__main__":
    main()
