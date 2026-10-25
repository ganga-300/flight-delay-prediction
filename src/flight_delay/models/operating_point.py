"""Post-selection calibration and operating-threshold utilities."""

import numpy as np
from sklearn.metrics import precision_recall_curve

from flight_delay.models.calibration import calibrate


def choose_threshold(y_true, probabilities, min_precision=0.60):
    """Maximum recall subject to the predeclared minimum precision."""
    if not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1]")

    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)

    if y.ndim != 1 or p.ndim != 1 or y.size == 0 or y.size != p.size:
        raise ValueError("Targets and probabilities must be equal-length vectors")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Targets must be binary")

    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")

    if np.sum(y == 1) == 0:
        raise ValueError("At least one positive target is required")

    precision, recall, thresholds = precision_recall_curve(y, p)

    # Final PR-curve point has no corresponding threshold.
    feasible = np.flatnonzero(precision[:-1] >= min_precision)

    if feasible.size == 0:
        return {
            "feasible": False,
            "threshold": None,
            "precision": None,
            "recall": None,
        }

    # Highest recall; ties use higher precision, then higher threshold.
    best = max(
        feasible,
        key=lambda i: (
            float(recall[i]),
            float(precision[i]),
            float(thresholds[i]),
        ),
    )

    return {
        "feasible": True,
        "threshold": float(thresholds[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
    }


def calibrate_and_select_threshold(
    fitted_model,
    X_calibrate,
    y_calibrate,
    X_threshold,
    y_threshold,
    feature_columns,
    min_precision=0.60,
):
    """Fit calibration on September, choose threshold on October."""
    cols = list(feature_columns)

    if len(cols) != 59 or len(set(cols)) != 59:
        raise ValueError("Exactly 59 unique frozen features are required")

    for frame in (X_calibrate, X_threshold):
        if set(frame.columns) != set(cols):
            raise ValueError("Unexpected input feature columns")

    calibrated = calibrate(
        fitted_model,
        X_calibrate[cols],
        y_calibrate,
        method="sigmoid",
    )

    probabilities = calibrated.predict_proba(X_threshold[cols])[:, 1]

    operating_point = choose_threshold(
        y_threshold,
        probabilities,
        min_precision=min_precision,
    )

    return {
        "calibrated_model": calibrated,
        "operating_point": operating_point,
    }
