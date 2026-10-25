"""Synthetic-data tests for XGBoost tuning."""

import numpy as np
import pandas as pd
import pytest

from flight_delay.models.tuning import tune_xgboost


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)

    numeric = [f"num_{i}" for i in range(49)]
    categorical = [f"cat_{i}" for i in range(10)]

    def make_frame(n):
        frame = pd.DataFrame(
            rng.normal(size=(n, 49)),
            columns=numeric,
        )

        for column in categorical:
            frame[column] = pd.Categorical(
                rng.choice(["A", "B"], size=n),
                categories=["A", "B"],
            )

        return frame

    X_fit = make_frame(100)
    X_tune = make_frame(40)

    y_fit = np.array([0, 1] * 50)
    y_tune = np.array([0, 1] * 20)

    return X_fit, y_fit, X_tune, y_tune, numeric, categorical


def test_xgboost_grid(sample_data):
    result = tune_xgboost(*sample_data)

    assert len(result["results"]) == 4
    assert result["best_model"] is not None
    assert result["best_params"]["max_depth"] in (3, 5)
    assert result["best_params"]["learning_rate"] in (0.03, 0.10)
    assert result["best_params"]["best_iteration"] >= 0
    assert 0 <= result["best_pr_auc"] <= 1


def test_rejects_missing_feature(sample_data):
    X_fit, y_fit, X_tune, y_tune, numeric, categorical = sample_data

    with pytest.raises(ValueError):
        tune_xgboost(
            X_fit.drop(columns=["num_0"]),
            y_fit,
            X_tune,
            y_tune,
            numeric,
            categorical,
        )


def test_rejects_single_class(sample_data):
    X_fit, y_fit, X_tune, _, numeric, categorical = sample_data

    with pytest.raises(ValueError):
        tune_xgboost(
            X_fit,
            y_fit,
            X_tune,
            np.zeros(len(X_tune), dtype=int),
            numeric,
            categorical,
        )
