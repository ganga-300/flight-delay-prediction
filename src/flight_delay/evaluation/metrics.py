"""Evaluation metrics for binary flight-delay prediction."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Average precision, used as PR-AUC."""
    return float(average_precision_score(y_true, y_prob))


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y_true, y_prob))


def recall_at_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float,
) -> tuple[float, float]:
    """Maximum recall and its threshold at minimum required precision."""
    if not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1]")

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    eligible = np.flatnonzero(precision[:-1] >= min_precision)

    if eligible.size == 0:
        return 0.0, float("nan")

    best_recall = np.max(recall[eligible])
    tied = eligible[np.isclose(recall[eligible], best_recall)]
    best = tied[-1]

    return float(recall[best]), float(thresholds[best])


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of predicted probabilities."""
    return float(brier_score_loss(y_true, y_prob))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Equal-width expected calibration error."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_true.size == 0 or y_true.size != y_prob.size:
        raise ValueError("Arrays must have equal nonzero length")

    if not np.all(np.isfinite(y_prob)) or np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")

    bins = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)

    error = 0.0
    for i in range(n_bins):
        mask = bins == i
        if mask.any():
            error += (
                mask.mean()
                * abs(y_true[mask].mean() - y_prob[mask].mean())
            )

    return float(error)
