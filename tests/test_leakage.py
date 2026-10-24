import pandas as pd
import pytest

from flight_delay.features.leakage import POST_DEPARTURE_COLUMNS, assert_no_leakage


def test_assert_no_leakage_rejects_post_departure_columns():
    with pytest.raises(ValueError):
        assert_no_leakage(["carrier", "dep_delay"])


def test_assert_no_leakage_accepts_clean_columns():
    assert_no_leakage(["carrier", "origin", "sched_dep_time"])


def test_feature_matrix_has_no_post_departure_or_target_columns(synthetic_raw, synthetic_config):
    from flight_delay.features.build import CATEGORICAL, build_features, numeric_features

    X = build_features(synthetic_raw, synthetic_config)
    feats = set(numeric_features(synthetic_config) + CATEGORICAL)
    assert feats <= set(X.columns)
    banned = POST_DEPARTURE_COLUMNS | {"is_delayed", "flight_status", "tailnum", "id"}
    assert not feats & banned
    assert set(X.columns) - feats == {"id", "date", "split", "is_delayed"}


# ---------------------------------------------------------------- focused leakage audit
POST = ["dep_time", "dep_delay", "arr_time", "arr_delay", "air_time"]
META = {"id", "date", "split", "is_delayed"}


def _scramble(raw, mask=None, seed=0):
    """Permute every post-departure field (keeping missingness) and overwrite tailnum."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out = raw.copy()
    mask = out.index == out.index if mask is None else mask
    for c in POST:
        m = mask & out[c].notna()
        out.loc[m, c] = rng.permutation(out.loc[m, c].to_numpy())
    out.loc[mask, "tailnum"] = "N0"
    return out


def _feature_frame(raw, cfg):
    from flight_delay.features.build import CATEGORICAL, build_feature_table, numeric_features

    t = build_feature_table(raw, cfg).set_index("id")
    return t[numeric_features(cfg) + CATEGORICAL].astype(str)


def test_sched_block_min_uses_only_scheduled_fields(synthetic_raw):
    from flight_delay.features.schedule import add_schedule_features

    base = add_schedule_features(synthetic_raw)["sched_block_min"]
    other = add_schedule_features(_scramble(synthetic_raw))["sched_block_min"]
    pd.testing.assert_series_equal(base, other)


def test_non_history_features_ignore_post_departure_fields_tailnum_and_id(
    synthetic_raw, synthetic_config
):
    from flight_delay.features.build import CATEGORICAL, numeric_features

    cols = [
        c for c in numeric_features(synthetic_config) + CATEGORICAL if not c.startswith("hist_")
    ]
    base = _feature_frame(synthetic_raw, synthetic_config)[cols]
    scr = _feature_frame(_scramble(synthetic_raw), synthetic_config)[cols]
    pd.testing.assert_frame_equal(base, scr.loc[base.index])
    remapped = synthetic_raw.assign(id=synthetic_raw["id"] * 7 + 1000)
    alt = _feature_frame(remapped, synthetic_config)
    alt.index = (alt.index - 1000) // 7
    pd.testing.assert_frame_equal(
        _feature_frame(synthetic_raw, synthetic_config), alt.loc[base.index]
    )


def test_no_feature_on_or_before_d_uses_post_departure_data_from_d_onward(
    synthetic_raw, synthetic_config
):
    """Scrambling every post-departure field of flights on/after D (including the flights
    being predicted on D) must not change ANY feature of flights on dates <= D."""
    cutoff = pd.Timestamp("2013-01-20")
    base = _feature_frame(synthetic_raw, synthetic_config)
    later = synthetic_raw["date"] >= cutoff
    scr = _feature_frame(_scramble(synthetic_raw, later), synthetic_config)
    keep = synthetic_raw.loc[synthetic_raw["date"] <= cutoff, "id"]
    keep = keep[keep.isin(base.index)]
    pd.testing.assert_frame_equal(base.loc[keep], scr.loc[keep])


def test_encoded_splits_hold_only_features_plus_metadata(synthetic_raw, synthetic_config):
    from flight_delay.features.build import (
        CATEGORICAL,
        build_feature_table,
        encode_splits,
        numeric_features,
    )

    feats = set(numeric_features(synthetic_config) + CATEGORICAL)
    splits, _ = encode_splits(
        build_feature_table(synthetic_raw, synthetic_config), synthetic_config
    )
    for frame in splits.values():
        assert set(frame.columns) == feats | META
        assert not POST_DEPARTURE_COLUMNS & set(frame.columns)
        assert frame[sorted(feats)].isna().sum().sum() == 0


def test_real_sched_block_matches_stdlib_reference(real_raw):
    """Independent zoneinfo implementation on all DST-switch dates plus a random sample."""
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo

    from flight_delay.features.airport_timezones import AIRPORT_TZ
    from flight_delay.features.schedule import add_schedule_features

    dst = real_raw["date"].isin(
        pd.to_datetime(["2013-03-09", "2013-03-10", "2013-11-02", "2013-11-03"])
    )
    sub = pd.concat([real_raw[dst], real_raw[~dst].sample(3000, random_state=0)])
    got = add_schedule_features(sub)["sched_block_min"]

    def ref(r, fold):
        z = ZoneInfo
        dep = datetime(
            r.year,
            r.month,
            r.day,
            r.sched_dep_time // 100,
            r.sched_dep_time % 100,
            tzinfo=z(AIRPORT_TZ[r.origin]),
            fold=fold,
        ).astimezone(timezone.utc)
        ah, am = divmod(int(r.sched_arr_time) % 2400, 100)
        best = None
        for k in (-1, 0, 1):
            d = datetime(r.year, r.month, r.day) + timedelta(days=k)
            arr = datetime(d.year, d.month, d.day, ah, am, tzinfo=z(AIRPORT_TZ[r.dest]), fold=fold)
            dur = (arr.astimezone(timezone.utc) - dep).total_seconds() / 60
            if dur >= 0 and (best is None or dur < best):
                best = dur
        return best

    for fold in (0, 1):  # both readings of a repeated (DST fall-back) hour agree for this data
        expected = pd.Series([ref(r, fold) for r in sub.itertuples()], index=sub.index)
        pd.testing.assert_series_equal(got, expected, check_names=False)


def test_real_no_scheduled_departure_in_dst_gap_or_overlap(real_raw):
    from flight_delay.features.airport_timezones import AIRPORT_TZ

    zones = real_raw["origin"].map(AIRPORT_TZ)
    for tz, idx in zones.groupby(zones).groups.items():
        loc = pd.DatetimeIndex(real_raw.loc[idx, "sched_dep_dt"]).tz_localize(
            tz, ambiguous="NaT", nonexistent="NaT"
        )
        assert not loc.isna().any(), tz


@pytest.mark.parametrize(
    "column",
    [
        "dep_time",
        "dep_delay",
        "arr_time",
        "arr_delay",
        "air_time",
        "id",
        "date",
        "split",
        "is_delayed",
        "flight_status",
        "tailnum",
    ],
)
def test_assert_no_leakage_rejects_all_forbidden_feature_columns(column):
    with pytest.raises(ValueError, match="Forbidden columns"):
        from flight_delay.features.leakage import assert_no_leakage

        assert_no_leakage(["carrier", column])


def test_assert_no_leakage_rejects_duplicate_features():
    from flight_delay.features.leakage import assert_no_leakage

    with pytest.raises(ValueError, match="Duplicate feature columns"):
        assert_no_leakage(["carrier", "carrier"])
