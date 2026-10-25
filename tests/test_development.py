"""Tests for leakage-safe chronological development splits."""

import pandas as pd
import pytest

from flight_delay.config import load_config
from flight_delay.features.build import CATEGORICAL, numeric_features
from flight_delay.models.development import prepare_development_splits


@pytest.fixture
def sample_table():
    dates = pd.to_datetime(
        [
            "2013-01-15",
            "2013-07-15",
            "2013-08-15",
            "2013-09-15",
            "2013-10-15",
        ]
    )
    cfg = load_config()
    data = {"date": dates}

    for col in numeric_features(cfg):
        data[col] = [1.0, 1.0, 999.0, 999.0, 999.0]

    for col in CATEGORICAL:
        data[col] = ["A", "A", "B", "C", "D"]

    return pd.DataFrame(data)


def test_period_sizes(sample_table):
    cfg = load_config()
    splits, _ = prepare_development_splits(sample_table, cfg)

    assert len(splits["fit"]) == 2
    assert len(splits["tune"]) == 1
    assert len(splits["calibrate"]) == 1
    assert len(splits["threshold"]) == 1


def test_encoder_medians_fit_only_on_jan_jul(sample_table):
    cfg = load_config()
    _, encoder = prepare_development_splits(sample_table, cfg)

    for col in numeric_features(cfg):
        assert encoder.medians_[col] == 1.0


def test_unseen_categories_become_other(sample_table):
    cfg = load_config()
    cfg["features"]["categorical_min_count"] = 1

    splits, _ = prepare_development_splits(sample_table, cfg)

    for col in CATEGORICAL:
        assert splits["tune"][col].iloc[0] == "__OTHER__"


def test_holdout_rows_are_rejected(sample_table):
    cfg = load_config()
    holdout_row = sample_table.iloc[[0]].copy()
    holdout_row["date"] = pd.Timestamp("2013-11-15")

    table = pd.concat([sample_table, holdout_row], ignore_index=True)

    with pytest.raises((ValueError, AssertionError)):
        prepare_development_splits(table, cfg)
