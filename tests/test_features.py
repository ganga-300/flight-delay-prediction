"""Feature-engineering tests, with emphasis on temporal leakage."""

import numpy as np
import pandas as pd
import pytest

from flight_delay.data.clean import build_labelled_frame
from flight_delay.features.build import (
    CATEGORICAL,
    build_feature_table,
    encode_splits,
    numeric_features,
)
from flight_delay.features.calendar import add_calendar_features, add_cyclic_encoding
from flight_delay.features.congestion import add_congestion_features
from flight_delay.features.encoding import OTHER, FeatureEncoder
from flight_delay.features.historical import add_historical_delay_rates, outcome_known_date
from flight_delay.features.schedule import add_schedule_features


def _features(table, cfg):
    return numeric_features(cfg) + CATEGORICAL


# ---------------------------------------------------------------- calendar / schedule
def test_cyclic_encoding_wraps():
    df = add_cyclic_encoding(pd.DataFrame({"h": [23.0, 0.0, 12.0]}), "h", 24)
    p = df[["h_sin", "h_cos"]].to_numpy()
    assert np.linalg.norm(p[0] - p[1]) < 0.3  # 23:00 is next to 00:00
    assert np.linalg.norm(p[1] - p[2]) == pytest.approx(2.0)  # midnight vs noon are opposite


def test_calendar_holidays_and_weekend():
    df = pd.DataFrame({"date": pd.to_datetime(["2013-07-04", "2013-07-06", "2013-12-31"])})
    df["sched_dep_dt"] = df["date"] + pd.Timedelta(hours=8)
    out = add_calendar_features(df)
    assert out["is_holiday"].tolist() == [1, 0, 0]
    assert out["is_weekend"].tolist() == [0, 1, 0]
    assert out["days_from_holiday"].tolist()[1] == 2  # two days after 4 July
    assert out["days_from_holiday"].tolist()[2] == -1  # day before New Year 2014 (padded range)
    assert out["sched_dep_hour_frac"].tolist() == [8.0, 8.0, 8.0]


def _sched(rows):
    df = pd.DataFrame(rows, columns=["date", "origin", "dest", "sched_dep_time", "sched_arr_time"])
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["sched_dep_time"] // 100
    df["sched_dep_dt"] = df["date"] + pd.to_timedelta(
        (df["sched_dep_time"] // 100) * 60 + df["sched_dep_time"] % 100, unit="m"
    )
    return add_schedule_features(df)


def test_scheduled_block_uses_common_time_basis():
    out = _sched(
        [
            ("2013-07-01", "JFK", "LAX", 2100, 5),  # PDT, arrives next local day
            ("2013-07-01", "JFK", "HNL", 900, 1430),  # HST (no DST): 6 h behind EDT
            ("2013-07-01", "JFK", "SJU", 2330, 300),  # AST = EDT in summer, overnight
            ("2013-11-03", "JFK", "LAX", 2040, 2359),  # DST ended 02:00 that day: PST vs EST
            ("2013-07-01", "EWR", "BOS", 600, 715),  # same zone
        ]
    )
    assert out["sched_block_min"].tolist() == [365.0, 690.0, 210.0, 379.0, 75.0]
    assert out["overnight_arrival"].tolist() == [1, 0, 1, 0, 0]


def test_scheduled_block_unknown_airport_is_nan():
    out = _sched([("2013-07-01", "JFK", "XXX", 900, 1100)])
    assert out["sched_block_min"].isna().all()


def test_real_data_time_basis_audit(real_raw):
    """With airport time zones, actual gate-to-gate minus air time (taxi) is never negative.
    Post-departure fields are used here only to validate the time-zone handling."""
    from flight_delay.features.timebase import arrival_utc, localize_to_utc

    lab = build_labelled_frame(real_raw)
    dep_local = lab["sched_dep_dt"] + pd.to_timedelta(lab["dep_delay"], unit="m")
    dep_u = localize_to_utc(dep_local, lab["origin"])
    arr_u = arrival_utc(dep_u, dep_local.dt.normalize(), lab["arr_time"], lab["dest"])
    taxi = (arr_u - dep_u).dt.total_seconds() / 60 - lab["air_time"]
    assert taxi.notna().all() and (taxi >= 0).all()
    block = add_schedule_features(real_raw)["sched_block_min"]
    assert block.notna().all() and block.between(1, 24 * 60).all()


# ---------------------------------------------------------------- congestion
def test_congestion_counts_use_full_raw_schedule(synthetic_raw, synthetic_config):
    cong = add_congestion_features(synthetic_raw)
    expected = synthetic_raw.groupby(["origin", "date", "hour"])["id"].transform("size")
    assert (cong["sched_dep_origin_hour"] == expected).all()
    # The conservative configuration excludes congestion features.
    conservative_config = {
        **synthetic_config,
        "features": {
            **synthetic_config.get("features", {}),
            "enabled_numeric_groups": ["calendar", "schedule_route"],
        },
    }
    table = build_feature_table(synthetic_raw, conservative_config)
    assert "sched_dep_origin_hour" not in table.columns

    # Enabling congestion explicitly should make the feature available.
    expanded_config = {
        **synthetic_config,
        "features": {
            **synthetic_config.get("features", {}),
            "enabled_numeric_groups": [
                "calendar", "schedule_route", "historical", "congestion"
            ],
        },
    }
    expanded_table = build_feature_table(synthetic_raw, expanded_config)
    assert "sched_dep_origin_hour" in expanded_table.columns

    labelled = build_labelled_frame(synthetic_raw)
    labelled_only = (
        labelled.groupby(["origin", "date", "hour"])["id"]
        .transform("size")
    )
    full_counts = cong.set_index("id")["sched_dep_origin_hour"]
    lab = labelled.assign(lab_n=labelled_only).set_index("id")

    assert (full_counts.loc[lab.index] >= lab["lab_n"]).all()
    assert (full_counts.loc[lab.index] > lab["lab_n"]).any()


# ---------------------------------------------------------------- historical: leakage
def _hist(df):
    lab = build_labelled_frame(df)
    lab["is_delayed"] = lab["is_delayed"].astype(int)
    return add_historical_delay_rates(lab, ["carrier", "origin"], [1, 7], prior_strength=5)


def test_first_date_has_no_history(synthetic_raw):
    h = _hist(synthetic_raw)
    first = h[h["date"] == h["date"].min()]
    assert first["hist_global_all_rate"].isna().all()
    assert (first["hist_carrier_7d_n"] == 0).all()


def test_unseen_key_falls_back_to_global_prior(synthetic_raw):
    raw = synthetic_raw.copy()
    raw.loc[raw.index[-1], "carrier"] = "ZZ"  # brand-new carrier on the last date
    h = _hist(raw)
    row = h[h["carrier"] == "ZZ"].iloc[0]
    assert row["hist_carrier_all_n"] == 0
    assert row["hist_carrier_all_rate"] == pytest.approx(row["hist_global_all_rate"])


def test_changing_same_day_and_future_labels_does_not_change_history(synthetic_raw):
    cutoff = pd.Timestamp("2013-01-20")
    base = _hist(synthetic_raw).set_index("id")
    flipped = synthetic_raw.copy()
    later = (flipped["date"] >= cutoff) & flipped["arr_delay"].notna()  # incl. the cutoff day
    flipped.loc[later, "arr_delay"] = np.where(flipped.loc[later, "arr_delay"] >= 15, -30.0, 300.0)
    alt = _hist(flipped).set_index("id")
    cols = [c for c in base.columns if c.startswith("hist_")]
    upto = base["date"] <= cutoff
    pd.testing.assert_frame_equal(base.loc[upto, cols], alt.loc[upto, cols])
    assert not base.loc[~upto, cols].equals(alt.loc[~upto, cols])  # sanity: later rows do change


def test_late_arriving_previous_day_flight_is_not_used_next_day():
    # 1 Jan 23:30 EWR->ORD, 60 min late, lands 02:00 CST = 03:00 EST on 2 Jan -> known 2 Jan.
    rows = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "carrier": ["UA"] * 3,
            "origin": ["EWR"] * 3,
            "dest": ["ORD"] * 3,
            "date": pd.to_datetime(["2013-01-01", "2013-01-02", "2013-01-03"]),
            "dep_delay": [60.0, 0.0, 0.0],
            "air_time": [120.0, 60.0, 60.0],
            "arr_time": [200.0, 930.0, 930.0],
            "is_delayed": [1, 0, 0],
        }
    )
    rows["sched_dep_dt"] = rows["date"] + pd.to_timedelta([23.5, 8, 8], unit="h")
    known = outcome_known_date(rows)
    assert known.iloc[0] == pd.Timestamp("2013-01-02")
    h = add_historical_delay_rates(rows, ["carrier"], [1])
    assert h.loc[1, "hist_carrier_1d_n"] == 0  # 2 Jan: the 1 Jan flight is not yet known
    # 3 Jan: both the overnight 1 Jan flight and the 2 Jan flight became known on 2 Jan.
    assert h.loc[2, "hist_carrier_1d_n"] == 2


def test_long_taxi_flight_is_known_only_after_actual_arrival():
    # Departs 21:00, 50 min airborne, but reaches the gate at 00:10 (3 h 20 min of taxi).
    # The old estimate (dep + air + 120 min = 23:50) would have counted it a day early.
    row = pd.DataFrame(
        {
            "origin": ["EWR"],
            "dest": ["BOS"],
            "date": pd.to_datetime(["2013-12-08"]),
            "dep_delay": [0.0],
            "air_time": [50.0],
            "arr_time": [10.0],
        }
    )
    row["sched_dep_dt"] = row["date"] + pd.Timedelta(hours=21)
    assert outcome_known_date(row).iloc[0] == pd.Timestamp("2013-12-09")


def test_real_data_outcomes_never_known_before_arrival(real_raw):
    from flight_delay.features.timebase import actual_arrival_ny

    lab = build_labelled_frame(real_raw)
    arrival_date = actual_arrival_ny(lab).dt.normalize()
    assert (outcome_known_date(lab) >= arrival_date).all()


def test_real_data_future_rows_cannot_change_earlier_features(real_raw):
    """Removing every record after a cutoff must leave all features up to the cutoff unchanged."""
    from flight_delay.config import load_config

    cfg = load_config()
    cutoff = pd.Timestamp("2013-06-15")
    full = build_feature_table(real_raw, cfg).set_index("id")
    trunc = build_feature_table(real_raw[real_raw["date"] <= cutoff], cfg).set_index("id")
    cols = _features(full, cfg)
    a = full.loc[trunc.index, cols].sort_index()
    b = trunc[cols].sort_index()
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_real_data_history_never_uses_same_or_later_dates(real_raw):
    lab = build_labelled_frame(real_raw)
    lab["is_delayed"] = lab["is_delayed"].astype(int)
    audit = add_historical_delay_rates(lab, ["carrier"], [1])
    # On each date the global past count must equal flights known strictly before that date.
    known = outcome_known_date(lab)
    for d in pd.to_datetime(["2013-01-02", "2013-05-01", "2013-12-31"]):
        expected = int((known < d).sum())
        got = audit.loc[audit["date"] == d, "hist_global_all_n"].unique()
        assert got.tolist() == [expected]


# ---------------------------------------------------------------- encoder: train-only fitting
def test_encoder_is_fitted_on_train_only(synthetic_raw, synthetic_config):
    table = build_feature_table(synthetic_raw, synthetic_config)
    splits, enc = encode_splits(table, synthetic_config)
    # Corrupting validation/test rows must not change what the encoder learned.
    noisy = table.copy()
    mask = noisy["split"] != "train"
    noisy.loc[mask, "carrier"] = "NEW"
    noisy.loc[mask, "distance"] = 1e9
    _, enc2 = encode_splits(noisy, synthetic_config)
    assert enc.to_dict() == enc2.to_dict()
    # Unseen levels map to __OTHER__, and categories are identical across splits.
    out = enc.transform(noisy[mask])
    assert (out["carrier"] == OTHER).all()
    cats = {name: list(s["carrier"].cat.categories) for name, s in splits.items()}
    assert cats["train"] == cats["valid"] == cats["test"]


def test_encoder_imputes_with_train_median_and_requires_fit():
    enc = FeatureEncoder(["c"], ["x"], min_count=2)
    with pytest.raises(RuntimeError):
        enc.transform(pd.DataFrame({"c": ["a"], "x": [1.0]}))
    train = pd.DataFrame({"c": ["a", "a", "b"], "x": [1.0, 3.0, np.nan]})
    enc.fit(train)
    out = enc.transform(pd.DataFrame({"c": ["a", "b", "q"], "x": [np.nan, 5.0, 0.0]}))
    assert out["x"].tolist() == [2.0, 5.0, 0.0]
    assert out["c"].astype(str).tolist() == ["a", OTHER, OTHER]  # b is rare (1 < 2), q unseen
    assert FeatureEncoder.from_dict(enc.to_dict()).to_dict() == enc.to_dict()


def test_build_is_deterministic(synthetic_raw, synthetic_config):
    a = build_feature_table(synthetic_raw, synthetic_config)
    b = build_feature_table(synthetic_raw.sample(frac=1, random_state=1), synthetic_config)
    pd.testing.assert_frame_equal(a, b)


def test_time_index_like_columns_are_not_features():
    from flight_delay.config import load_config
    from flight_delay.features.build import feature_groups

    cfg = load_config()
    feats = set(numeric_features(cfg))
    month_like = {"month", "day_of_year", "month_sin", "month_cos", "doy_sin", "doy_cos"}
    assert not month_like & feats  # unseen months in valid/test (chronological split)
    assert not [c for c in feats if c.endswith("_all_n")]
    assert {"dow_sin", "is_holiday", "sched_block_min"} <= feats

    # Expanded features are allowed only when their groups are enabled.
    groups = feature_groups(cfg)
    enabled_groups = cfg.get("features", {}).get(
        "enabled_numeric_groups",
        ["calendar", "schedule_route", "congestion", "historical"],
    )
    if isinstance(enabled_groups, dict):
        enabled_groups = [
            name for name, enabled in enabled_groups.items() if enabled
        ]

    if "historical" in enabled_groups:
        assert set(groups["historical"]) & feats
    else:
        assert not set(groups["historical"]) & feats

    if "congestion" in enabled_groups:
        assert set(groups["congestion"]) & feats
    else:
        assert not set(groups["congestion"]) & feats

    assert "sched_clock_duration_min" not in feats


def test_missing_arrival_time_uses_conservative_fallback():
    from flight_delay.features.historical import outcome_known_date

    row = pd.DataFrame({
        "origin": ["EWR"],
        "dest": ["BOS"],
        "date": pd.to_datetime(["2013-01-01"]),
        "sched_dep_dt": pd.to_datetime(["2013-01-01 23:00"]),
        "dep_delay": [60.0],
        "air_time": [60.0],
        "arr_time": [float("nan")],
    })

    # 23:00 + 60 min delay + 60 min air time + 240 min buffer = 06:00 next day.
    assert outcome_known_date(row).iloc[0] == pd.Timestamp("2013-01-02")


def test_unknown_airport_timezone_uses_conservative_date_fallback():
    from flight_delay.features.historical import outcome_known_date

    row = pd.DataFrame({
        "origin": ["XXX"],
        "dest": ["YYY"],
        "date": pd.to_datetime(["2013-01-01"]),
        "sched_dep_dt": pd.to_datetime(["2013-01-01 23:00"]),
        "dep_delay": [60.0],
        "air_time": [60.0],
        "arr_time": [float("nan")],
    })

    # Without an airport timezone, use the conservative date + 2 days.
    assert outcome_known_date(row).iloc[0] == pd.Timestamp("2013-01-03")




@pytest.mark.parametrize(
    ("window", "expected_count"),
    [(1, 1), (7, 2), (28, 3)],
)
def test_history_windows_use_exact_prior_calendar_dates(window, expected_count):
    """Windows include D-w through D-1 and exclude the prediction date D."""
    from flight_delay.features.historical import _past_window_sums

    daily = pd.DataFrame(
        {"count": [1, 1, 1]},
        index=pd.to_datetime([
            "2013-01-01",
            "2013-01-02",
            "2013-01-08",
        ]),
    )
    dates = pd.date_range("2013-01-01", "2013-01-30", freq="D")
    result = _past_window_sums(daily, dates, window)

    assert result.loc[pd.Timestamp("2013-01-09"), "count"] == expected_count


def test_missing_arrival_fallback_uses_new_york_calendar_date():
    from flight_delay.features.historical import outcome_known_date

    row = pd.DataFrame({
        "origin": ["LAX"],
        "dest": ["JFK"],
        "date": pd.to_datetime(["2013-01-01"]),
        "sched_dep_dt": pd.to_datetime(["2013-01-01 23:00"]),
        "dep_delay": [0.0],
        "air_time": [60.0],
        "arr_time": [float("nan")],
    })

    # LAX departure: Jan 1, 23:00 PST = Jan 2, 07:00 UTC.
    # Add 60 min air time and 240 min buffer:
    # Jan 2, 12:00 UTC = Jan 2, 07:00 EST.
    assert (
        outcome_known_date(row).iloc[0]
        == pd.Timestamp("2013-01-02")
    )


def test_outcome_known_on_date_is_excluded_from_that_dates_history():
    rows = pd.DataFrame({
        "carrier": ["UA", "UA", "UA"],
        "origin": ["EWR", "EWR", "EWR"],
        "dest": ["BOS", "BOS", "BOS"],
        "date": pd.to_datetime([
            "2013-01-01", "2013-01-02", "2013-01-03"
        ]),
        "sched_dep_dt": pd.to_datetime([
            "2013-01-01 08:00",
            "2013-01-02 08:00",
            "2013-01-03 08:00",
        ]),
        "dep_delay": [0.0, 0.0, 0.0],
        "air_time": [60.0, 60.0, 60.0],
        "arr_time": [100.0, 100.0, 100.0],
        "is_delayed": [1, 0, 0],
    })

    known = outcome_known_date(rows)
    assert known.iloc[0] == pd.Timestamp("2013-01-02")

    result = add_historical_delay_rates(
        rows,
        keys=["carrier"],
        windows_days=[1],
        include_all_past=True,
    )

    assert result.loc[0, "hist_global_all_n"] == 0
    assert result.loc[0, "hist_carrier_1d_n"] == 0
    # The outcome becomes known on Jan 2, so it is excluded on Jan 2
    # and included in all-past history on Jan 3.
    assert result.loc[1, "hist_global_all_n"] == 0
    assert result.loc[1, "hist_carrier_1d_n"] == 0
    assert result.loc[2, "hist_global_all_n"] == 1
    assert result.loc[2, "hist_carrier_1d_n"] == 1
