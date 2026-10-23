import numpy as np
import pytest

from flight_delay.evaluation.metrics import (
    expected_calibration_error,
    recall_at_precision,
)


def test_recall_at_precision_known_case():
    """Check recall and threshold for a hand-computed example."""
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.1])

    recall, threshold = recall_at_precision(
        y_true, y_prob, min_precision=1.0
    )

    assert recall == pytest.approx(1.0)
    assert threshold == pytest.approx(0.8)


def test_ece_is_zero_for_perfect_calibration():
    """ECE should be zero when confidence matches observed outcomes."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])

    ece = expected_calibration_error(
        y_true, y_prob, n_bins=10
    )

    assert ece == pytest.approx(0.0)


def test_recall_at_precision_returns_zero_when_target_unreachable():
    """Return zero recall and infinity when no threshold meets the target."""
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.8, 0.9, 0.6, 0.7])

    recall, threshold = recall_at_precision(
        y_true, y_prob, min_precision=1.0
    )

    assert recall == pytest.approx(0.0)
    assert np.isinf(threshold)
