"""Synthetic integration tests for the D-201 workflow."""

import numpy as np
import pandas as pd
import pytest

from flight_delay.models.workflow import run_modeling_workflow


def synthetic_splits():
    rng = np.random.default_rng(42)
    numeric = [f"num_{i}" for i in range(49)]
    categorical = [f"cat_{i}" for i in range(10)]
    splits = {}

    for period, size in (
        ("fit", 100),
        ("tune", 40),
        ("calibrate", 40),
        ("threshold", 40),
    ):
        frame = pd.DataFrame(
            rng.normal(size=(size, 49)),
            columns=numeric,
        )

        for col in categorical:
            frame[col] = pd.Categorical(
                rng.choice(["A", "B"], size=size),
                categories=["A", "B"],
            )

        frame["is_delayed"] = np.array([0, 1] * (size // 2))
        splits[period] = frame

    return splits, numeric, categorical


def test_end_to_end_workflow():
    splits, numeric, categorical = synthetic_splits()
    splits = {"fit": splits["fit"], "tune": splits["tune"]}

    result = run_modeling_workflow(
        splits, numeric, categorical
    )

    assert len(result["logistic"]["results"]) == 8
    assert len(result["xgboost"]["results"]) == 4
    assert result["selection"]["model_name"] in ("logreg", "xgboost")
    assert "calibrated_model" not in result
    assert "operating_point" not in result


def test_rejects_missing_period():
    splits, numeric, categorical = synthetic_splits()
    splits = {"fit": splits["fit"], "tune": splits["tune"]}
    del splits["tune"]

    with pytest.raises(ValueError):
        run_modeling_workflow(splits, numeric, categorical)


def test_rejects_missing_feature():
    splits, numeric, categorical = synthetic_splits()
    splits = {"fit": splits["fit"], "tune": splits["tune"]}
    splits["fit"] = splits["fit"].drop(columns=["num_0"])

    with pytest.raises(ValueError):
        run_modeling_workflow(splits, numeric, categorical)
