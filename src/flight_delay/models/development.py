"""Chronological development splits with leakage-safe preprocessing."""

import pandas as pd

from flight_delay.data.split import assert_no_holdout
from flight_delay.features.build import CATEGORICAL, numeric_features
from flight_delay.features.encoding import FeatureEncoder


PERIODS = {
    "fit": ("2013-01-01", "2013-07-31"),
    "tune": ("2013-08-01", "2013-08-31"),
    "calibrate": ("2013-09-01", "2013-09-30"),
    "threshold": ("2013-10-01", "2013-10-31"),
}


def prepare_development_splits(table: pd.DataFrame, config: dict):
    """Fit the encoder only on Jan-Jul and transform later periods."""
    dates = pd.to_datetime(table["date"])
    assert_no_holdout(dates, config["split"])

    expected = set(numeric_features(config) + CATEGORICAL)
    if len(expected) != 59:
        raise ValueError(f"Expected 59 frozen features, got {len(expected)}")

    raw_splits = {}

    for name, (start, end) in PERIODS.items():
        mask = dates.between(pd.Timestamp(start), pd.Timestamp(end))
        raw_splits[name] = table.loc[mask].copy()

        if raw_splits[name].empty:
            raise ValueError(f"Empty development period: {name}")

    encoder = FeatureEncoder(
        categorical=CATEGORICAL,
        numeric=numeric_features(config),
        min_count=config["features"]["categorical_min_count"],
    )

    encoder.fit(raw_splits["fit"])

    encoded = {
        name: encoder.transform(frame)
        for name, frame in raw_splits.items()
    }

    return encoded, encoder
