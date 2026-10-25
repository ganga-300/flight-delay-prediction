"""Behaviour the D-203 evaluation plan relies on (Member 4 review of inherited code)."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from flight_delay.evaluation.metrics import expected_calibration_error, recall_at_precision
from flight_delay.models.calibration import calibrate
from flight_delay.models.operating_point import choose_threshold


@pytest.fixture
def balanced_model_and_data():
    """A class-weight-balanced model, like the D-202 final model, whose raw scores run high."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6000, 3))
    y = (X[:, 0] + rng.normal(size=6000) > 1).astype(int)
    model = LogisticRegression(class_weight="balanced").fit(X[:3000], y[:3000])
    return model, X, y


def test_calibration_fits_one_sigmoid_on_all_calibration_rows(balanced_model_and_data):
    model, X, y = balanced_model_and_data
    calibrated = calibrate(model, X[3000:4500], y[3000:4500], method="sigmoid")
    assert len(calibrated.calibrated_classifiers_) == 1


def test_calibration_corrects_balanced_class_weight_scores(balanced_model_and_data):
    model, X, y = balanced_model_and_data
    calibrated = calibrate(model, X[3000:4500], y[3000:4500], method="sigmoid")
    raw = model.predict_proba(X[4500:])[:, 1]
    cal = calibrated.predict_proba(X[4500:])[:, 1]
    assert expected_calibration_error(y[4500:], cal) < expected_calibration_error(y[4500:], raw)


@pytest.mark.parametrize("min_precision", [0.5, 0.6, 0.7])
def test_flag_rule_reproduces_the_chosen_operating_point(min_precision):
    """Flagging p >= threshold gives exactly the precision and recall choose_threshold reports."""
    for seed in range(50):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, 400)
        p = np.round(rng.random(400), 2)
        op = choose_threshold(y, p, min_precision)
        if not op["feasible"]:
            continue
        flagged = p >= op["threshold"]
        assert y[flagged].mean() == pytest.approx(op["precision"])
        assert flagged[y == 1].mean() == pytest.approx(op["recall"])
        assert op["precision"] >= min_precision


@pytest.mark.parametrize("min_precision", [0.5, 0.6, 0.7])
def test_threshold_functions_agree(min_precision):
    for seed in range(50):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, 400)
        p = np.round(rng.random(400), 2)
        op = choose_threshold(y, p, min_precision)
        recall, threshold = recall_at_precision(y, p, min_precision)
        if op["feasible"]:
            assert (recall, threshold) == pytest.approx((op["recall"], op["threshold"]))
        else:
            assert recall == 0.0 and np.isnan(threshold)
