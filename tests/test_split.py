import pandas as pd
import pytest

from flight_delay.data.split import assign_split, chronological_split


def test_split_is_chronological_and_disjoint(synthetic_raw, synthetic_config):
    parts = chronological_split(synthetic_raw, "date", synthetic_config["split"])
    assert parts["train"]["date"].max() < parts["valid"]["date"].min()
    assert parts["valid"]["date"].max() < parts["test"]["date"].min()
    ids = [set(p["id"]) for p in parts.values()]
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2]) and not (ids[0] & ids[2])
    assert sum(len(p) for p in parts.values()) == len(synthetic_raw)  # 40 days fully covered


def test_split_rejects_overlap_and_disorder():
    d = pd.Series(pd.to_datetime(["2013-01-01"]))
    with pytest.raises(ValueError):
        assign_split(
            d, {"train": ["2013-01-01", "2013-02-01"], "valid": ["2013-01-15", "2013-03-01"]}
        )
    with pytest.raises(ValueError):
        assign_split(d, {"train": ["2013-02-01", "2013-01-01"]})


def test_config_split_matches_existing_dates(real_raw):
    from flight_delay.config import load_config

    cfg = load_config()
    s = assign_split(real_raw["date"], cfg["split"])
    assert s.notna().all()
    assert real_raw.loc[s == "train", "date"].max() == pd.Timestamp("2013-08-31")
    assert real_raw.loc[s == "valid", "date"].min() == pd.Timestamp("2013-09-01")
    assert real_raw.loc[s == "valid", "date"].max() == pd.Timestamp("2013-10-31")
    # The development data holds no test rows; the test period itself is still configured.
    assert not (s == "test").any()
    assert cfg["split"]["test"] == ["2013-11-01", "2013-12-31"]
