"""Tests for pre-registered Logistic Regression tuning."""

import numpy as np
import pandas as pd
import pytest

from flight_delay.models.tuning import tune_logistic


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)

    columns = [f"num_{i}" for i in range(49)]
    categories = [f"cat_{i}" for i in range(10)]

    def make_frame(n):
        frame = pd.DataFrame(
            rng.normal(size=(n, 49)),
            columns=columns,
        )
        for name in categories:
            frame[name] = rng.choice(["A", "B"], size=n)
        return frame

    X_fit = make_frame(100)
    X_tune = make_frame(40)

    y_fit = np.array([0, 1] * 50)
    y_tune = np.array([0, 1] * 20)

    return X_fit, y_fit, X_tune, y_tune, columns, categories


def test_logistic_grid(sample_data):
    result = tune_logistic(*sample_data)

    assert len(result["results"]) == 8
    assert result["best_model"] is not None
    assert result["best_params"]["C"] in (0.01, 0.1, 1.0, 10.0)
    assert result["best_params"]["class_weight"] in (None, "balanced")
    assert 0 <= result["best_pr_auc"] <= 1


def test_rejects_missing_feature(sample_data):
    X_fit, y_fit, X_tune, y_tune, numeric, categorical = sample_data

    with pytest.raises(ValueError):
        tune_logistic(
            X_fit.drop(columns=["num_0"]),
            y_fit,
            X_tune,
            y_tune,
            numeric,
            categorical,
        )


def test_rejects_incorrect_feature_count(sample_data):
    X_fit, y_fit, X_tune, y_tune, numeric, categorical = sample_data

    with pytest.raises(ValueError):
        tune_logistic(
            X_fit,
            y_fit,
            X_tune,
            y_tune,
            numeric[:-1],
            categorical,
        )
