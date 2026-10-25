"""Tests for binary classification evaluation metrics."""

import numpy as np
import pytest

from flight_delay.evaluation.metrics import (
    brier,
    expected_calibration_error,
    pr_auc,
    recall_at_precision,
    roc_auc,
)


def test_perfect_classification():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])

    assert pr_auc(y, p) == pytest.approx(1.0)
    assert roc_auc(y, p) == pytest.approx(1.0)
    assert brier(y, p) == pytest.approx(0.025)


def test_recall_at_precision_known_case():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1])

    recall, threshold = recall_at_precision(y, p, 0.60)

    assert recall == pytest.approx(1.0)
    assert threshold == pytest.approx(0.7)


def test_recall_at_precision_invalid_target():
    y = np.array([0, 1])
    p = np.array([0.2, 0.8])

    with pytest.raises(ValueError):
        recall_at_precision(y, p, 0.0)


def test_ece_is_zero_for_perfect_calibration():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])

    assert expected_calibration_error(y, p) == pytest.approx(0.0)


def test_ece_known_case():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])

    assert expected_calibration_error(y, p) == pytest.approx(0.15)


def test_ece_invalid_probability():
    y = np.array([0, 1])
    p = np.array([0.2, 1.2])

    with pytest.raises(ValueError):
        expected_calibration_error(y, p)
