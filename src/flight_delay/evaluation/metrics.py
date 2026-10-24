"""Evaluation metrics required by the project brief."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return area under the precision-recall curve (average precision)."""
    return float(average_precision_score(y_true, y_prob))


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the area under the ROC curve."""
    return float(roc_auc_score(y_true, y_prob))


def recall_at_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float,
) -> tuple[float, float]:
    """Return (best recall, threshold) among thresholds meeting min_precision.

    Select the threshold on validation data and apply it unchanged to test data.
    If no threshold satisfies the requested precision, return (0.0, inf).
    """
    if not 0.0 <= min_precision <= 1.0:
        raise ValueError("min_precision must be between 0 and 1.")

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # precision and recall contain one extra terminal element compared
    # with thresholds, so exclude that final element when pairing them.
    precision = precision[:-1]
    recall = recall[:-1]

    eligible = np.flatnonzero(precision >= min_precision)

    if eligible.size == 0:
        return 0.0, float("inf")

    best_recall = np.max(recall[eligible])
    best_candidates = eligible[recall[eligible] == best_recall]

    # When recall ties, prefer the higher threshold.
    best_idx = best_candidates[np.argmax(thresholds[best_candidates])]

    return float(recall[best_idx]), float(thresholds[best_idx])


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the Brier score; lower is better."""
    return float(brier_score_loss(y_true, y_prob))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate expected calibration error using equal-width probability bins."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("y_true and y_prob must be one-dimensional.")

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have equal lengths.")

    if len(y_true) == 0:
        raise ValueError("y_true and y_prob must not be empty.")

    if not np.all(np.isfinite(y_prob)):
        raise ValueError("y_prob must contain only finite values.")

    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    # Equal-width bins: [0, 1/n), ..., [(n-1)/n, 1].
    bin_ids = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)

    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx

        if not np.any(mask):
            continue

        mean_confidence = np.mean(y_prob[mask])
        fraction_positive = np.mean(y_true[mask])

        ece += (np.sum(mask) / len(y_true)) * abs(
            fraction_positive - mean_confidence
        )

    return float(ece)
