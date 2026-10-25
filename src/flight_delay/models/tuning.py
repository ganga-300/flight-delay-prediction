"""Pre-registered, validation-only model selection utilities."""

from itertools import product

import numpy as np
from sklearn.metrics import average_precision_score

from flight_delay.models.logistic import build_logistic_pipeline

LOGREG_C_VALUES = (0.01, 0.1, 1.0, 10.0)
LOGREG_WEIGHTS = (None, "balanced")


def tune_logistic(X_fit, y_fit, X_tune, y_tune, numeric_cols, categorical_cols):
    """Fit on Jan-Jul and select using August PR-AUC only."""

    expected = list(numeric_cols) + list(categorical_cols)

    if len(expected) != 59 or len(set(expected)) != 59:
        raise ValueError("Exactly 59 unique frozen features are required")

    if set(X_fit.columns) != set(expected):
        raise ValueError("Training feature columns do not match frozen features")

    if set(X_tune.columns) != set(expected):
        raise ValueError("Validation feature columns do not match frozen features")

    if len(X_fit) != len(y_fit) or len(X_tune) != len(y_tune):
        raise ValueError("Feature and target lengths must match")

    results = []
    best_model = None
    best_score = -np.inf
    best_params = None

    for C, class_weight in product(LOGREG_C_VALUES, LOGREG_WEIGHTS):
        params = {
            "C": C,
            "class_weight": class_weight,
            "max_iter": 2000,
        }

        model = build_logistic_pipeline(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            params=params,
        )

        model.fit(X_fit[expected], y_fit)

        probabilities = model.predict_proba(X_tune[expected])[:, 1]
        score = float(average_precision_score(y_tune, probabilities))

        results.append({
            "model": "logreg",
            "C": C,
            "class_weight": class_weight,
            "pr_auc": score,
        })

        if score > best_score + 1e-12:
            best_score = score
            best_model = model
            best_params = params.copy()

    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_pr_auc": best_score,
        "results": results,
    }


XGBOOST_DEPTHS = (3, 5)
XGBOOST_LEARNING_RATES = (0.03, 0.10)


def tune_xgboost(
    X_fit,
    y_fit,
    X_tune,
    y_tune,
    numeric_cols,
    categorical_cols,
):
    """Tune XGBoost on the pre-registered August validation period."""
    from flight_delay.models.xgboost_model import build_xgboost

    expected = list(numeric_cols) + list(categorical_cols)

    if len(expected) != 59 or len(set(expected)) != 59:
        raise ValueError("Exactly 59 unique frozen features are required")

    if set(X_fit.columns) != set(expected):
        raise ValueError("Training features do not match frozen features")

    if set(X_tune.columns) != set(expected):
        raise ValueError("Validation features do not match frozen features")

    if len(X_fit) != len(y_fit) or len(X_tune) != len(y_tune):
        raise ValueError("Features and targets must have matching lengths")

    if len(set(y_fit)) != 2 or len(set(y_tune)) != 2:
        raise ValueError("Both periods must contain both target classes")

    results = []
    best_model = None
    best_params = None
    best_score = -np.inf

    for max_depth, learning_rate in product(
        XGBOOST_DEPTHS,
        XGBOOST_LEARNING_RATES,
    ):
        params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": 1000,
            "early_stopping_rounds": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

        model = build_xgboost(params=params)

        model.fit(
            X_fit[expected],
            y_fit,
            eval_set=[(X_tune[expected], y_tune)],
            verbose=False,
        )

        probabilities = model.predict_proba(
            X_tune[expected],
        )[:, 1]

        score = float(
            average_precision_score(y_tune, probabilities)
        )

        results.append({
            "model": "xgboost",
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "best_iteration": int(model.best_iteration),
            "pr_auc": score,
        })

        if score > best_score + 1e-12:
            best_score = score
            best_model = model
            best_params = params.copy()
            best_params["best_iteration"] = int(model.best_iteration)

    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_pr_auc": best_score,
        "results": results,
    }
