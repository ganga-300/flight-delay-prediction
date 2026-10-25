"""Persist development-only experiment results and selected model."""

import json
from pathlib import Path

from flight_delay.models.persistence import save_model


def save_experiment_artifacts(result, encoder, output_dir):
    """Save model, encoder, tuning grids and operating threshold."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    selected = result["selection"]

    if selected["model_name"] not in {"logreg", "xgboost"}:
        raise ValueError("Unknown selected model")

    if len(result["logistic"]["results"]) != 8:
        raise ValueError("Expected eight Logistic Regression experiments")

    if len(result["xgboost"]["results"]) != 4:
        raise ValueError("Expected four XGBoost experiments")

    model_path = destination / "selected_model.joblib"
    encoder_path = destination / "encoder.json"
    results_path = destination / "development_results.json"

    # The calibrated model includes the frozen fitted base model.
    save_model(result["calibrated_model"], model_path)
    encoder.save(encoder_path)

    report = {
        "status": "development_only",
        "protocol": "D-201",
        "test_evaluated": False,
        "feature_count": len(encoder.numeric) + len(encoder.categorical),
        "selected_model": selected["model_name"],
        "selected_params": selected["parameters"],
        "validation_pr_auc": selected["validation_pr_auc"],
        "model_comparison": selected["comparison"],
        "logistic_experiments": result["logistic"]["results"],
        "xgboost_experiments": result["xgboost"]["results"],
        "operating_point": result["operating_point"],
    }

    if report["feature_count"] != 59:
        raise ValueError("Exactly 59 frozen features are required")

    results_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )

    return {
        "model": model_path,
        "encoder": encoder_path,
        "results": results_path,
    }
