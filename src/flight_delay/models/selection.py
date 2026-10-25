"""Deterministic model selection from development validation results."""

import math


def select_best_model(logistic_result: dict, xgboost_result: dict):
    """Select the winner using validation PR-AUC only."""

    candidates = [
        ("logreg", logistic_result),
        ("xgboost", xgboost_result),
    ]

    for name, result in candidates:
        if result.get("best_model") is None:
            raise ValueError(f"{name}: missing fitted model")

        score = result.get("best_pr_auc")

        if not isinstance(score, (int, float)):
            raise ValueError(f"{name}: invalid PR-AUC")

        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"{name}: PR-AUC must be between 0 and 1")

    # D-201 tie-break: prefer the simpler model (Logistic Regression).
    winner_name, winner_result = max(
        candidates,
        key=lambda item: (
            item[1]["best_pr_auc"],
            item[0] == "logreg",
        ),
    )

    return {
        "model_name": winner_name,
        "model": winner_result["best_model"],
        "parameters": winner_result["best_params"],
        "validation_pr_auc": winner_result["best_pr_auc"],
        "comparison": {
            name: result["best_pr_auc"]
            for name, result in candidates
        },
    }
