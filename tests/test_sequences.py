"""Daily delay sequences for the Member 2 GRU (D-303): leakage, alignment and scaling."""

import numpy as np
import pandas as pd
import pytest

from flight_delay.data.clean import build_labelled_frame
from flight_delay.features.historical import add_historical_delay_rates, outcome_known_date
from flight_delay.features.sequences import (
    SEQUENCE_CHANNELS,
    UNKNOWN,
    SequenceScaler,
    build_sequence_tables,
    gather_sequences,
)


def _labelled(raw):
    lab = build_labelled_frame(raw)
    lab["is_delayed"] = lab["is_delayed"].astype(int)
    return lab.sort_values(["date", "id"]).reset_index(drop=True)


def _sequences(lab, length=7):
    tables = build_sequence_tables(lab, lab["date"].max(), max_len=28)
    idx = tables.flight_indices(lab)
    return tables, idx, gather_sequences(tables.daily_arrays(), idx, length)


def test_shape_and_channel_order(synthetic_raw):
    lab = _labelled(synthetic_raw)
    _, _, seq = _sequences(lab, length=7)
    assert seq.shape == (len(lab), 7, len(SEQUENCE_CHANNELS))
    assert seq.dtype == np.float32


def test_changing_same_day_and_future_labels_does_not_change_sequences(synthetic_raw):
    cutoff = pd.Timestamp("2013-01-20")
    base_lab = _labelled(synthetic_raw)
    flipped = synthetic_raw.copy()
    later = (flipped["date"] >= cutoff) & flipped["arr_delay"].notna()  # incl. the cutoff day
    flipped.loc[later, "arr_delay"] = np.where(flipped.loc[later, "arr_delay"] >= 15, -30.0, 300.0)
    alt_lab = _labelled(flipped)
    _, _, base = _sequences(base_lab, length=14)
    _, _, alt = _sequences(alt_lab, length=14)
    upto = (base_lab["date"] <= cutoff).to_numpy()
    np.testing.assert_array_equal(base[upto], alt[upto])  # NaN padding compares equal
    assert not np.array_equal(base[~upto], alt[~upto], equal_nan=True)  # later rows do change


def test_late_arriving_previous_day_flight_is_not_in_next_days_sequence():
    # 1 Jan 23:30 EWR->ORD, 60 min late, lands 03:00 EST on 2 Jan -> known 2 Jan.
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
    assert outcome_known_date(rows).iloc[0] == pd.Timestamp("2013-01-02")
    tables = build_sequence_tables(rows, rows["date"].max(), max_len=3)
    idx = tables.flight_indices(rows)
    n_carrier = tables.counts["carrier"][0]
    last_step = idx["day"] - 1  # D-1
    carrier_n = n_carrier[last_step, idx["carrier"]]
    assert carrier_n.tolist() == [0, 0, 2]  # 2 Jan sees nothing from 1 Jan; 3 Jan sees both


def test_window_counts_match_phase1_history_features(synthetic_raw):
    lab = _labelled(synthetic_raw)
    hist = add_historical_delay_rates(lab, ["carrier", "origin", "dest"], [1, 7])
    tables, idx, _ = _sequences(lab)
    for entity in ("carrier", "origin", "dest"):
        n = tables.counts[entity][0]
        for w in (1, 7):
            steps = idx["day"][:, None] + np.arange(-w, 0)[None, :]
            window = n[steps, idx[entity][:, None]].sum(1)
            np.testing.assert_array_equal(window, hist[f"hist_{entity}_{w}d_n"].to_numpy())


def test_last_step_is_previous_day(synthetic_raw):
    lab = _labelled(synthetic_raw)
    tables, idx, seq = _sequences(lab, length=3)
    prev = lab["date"] - pd.Timedelta(days=1)
    dow = prev.dt.day_of_week.to_numpy()
    sin_col = SEQUENCE_CHANNELS.index("dow_sin")
    np.testing.assert_allclose(seq[:, -1, sin_col], np.sin(2 * np.pi * dow / 7), atol=1e-6)
    system_n = tables.counts["system"][0][:, 0]
    np.testing.assert_allclose(seq[:, -1, 1], np.log1p(system_n[idx["day"] - 1]), rtol=1e-6)


def test_first_day_is_all_padding(synthetic_raw):
    lab = _labelled(synthetic_raw)
    _, _, seq = _sequences(lab, length=7)
    first = (lab["date"] == lab["date"].min()).to_numpy()
    rate_cols = [SEQUENCE_CHANNELS.index(c) for c in SEQUENCE_CHANNELS if c.endswith("_rate")]
    count_cols = [SEQUENCE_CHANNELS.index(c) for c in SEQUENCE_CHANNELS if "log_n" in c]
    assert np.isnan(seq[first][:, :, rate_cols]).all()
    assert (seq[first][:, :, count_cols] == 0).all()
    assert (seq[first][:, :, SEQUENCE_CHANNELS.index("carrier_has_data")] == 0).all()


def test_unseen_level_maps_to_unknown_and_global_rate(synthetic_raw):
    lab = _labelled(synthetic_raw)
    tables = build_sequence_tables(lab, lab["date"].max())
    row = lab.iloc[[-1]].copy()
    row["carrier"] = "ZZ"
    idx = tables.flight_indices(row)
    assert tables.levels["carrier"][idx["carrier"][0]] == UNKNOWN
    seq = gather_sequences(tables.daily_arrays(), idx, 5)
    carrier_rate = seq[0, :, SEQUENCE_CHANNELS.index("carrier_rate")]
    system_n = tables.counts["system"][0][:, 0]
    steps = idx["day"][0] + np.arange(-5, 0)
    with np.errstate(invalid="ignore"):
        prior = np.cumsum(tables.counts["system"][1][:, 0]) / np.cumsum(system_n)
    np.testing.assert_allclose(carrier_rate, prior[steps], rtol=1e-6)  # n = 0 -> prior


def test_scaler_uses_only_training_rows_and_maps_nan_to_zero(synthetic_raw):
    lab = _labelled(synthetic_raw)
    tables = build_sequence_tables(lab, lab["date"].max())
    idx = tables.flight_indices(lab)
    train = (lab["date"] <= "2013-01-25").to_numpy()
    train_idx = {k: v[train] for k, v in idx.items()}
    scaler = SequenceScaler.fit(tables, train_idx, 7)

    raw = gather_sequences(tables.daily_arrays(), train_idx, 7).reshape(-1, len(SEQUENCE_CHANNELS))
    raw = raw.astype(np.float64)
    np.testing.assert_allclose(scaler.mean, np.nanmean(raw, axis=0), rtol=1e-5, atol=1e-6)

    scaled = gather_sequences(scaler.transform_tables(tables), idx, 7)
    assert np.isfinite(scaled).all()
    first = (lab["date"] == lab["date"].min()).to_numpy()
    assert (scaled[first][:, :, SEQUENCE_CHANNELS.index("system_rate")] == 0).all()


def test_dates_in_padding_or_outside_tables_are_rejected(synthetic_raw):
    lab = _labelled(synthetic_raw)
    tables = build_sequence_tables(lab, lab["date"].max())
    with pytest.raises(ValueError):
        tables.day_index(pd.Series(pd.to_datetime(["2012-12-31"])))
    with pytest.raises(ValueError):
        tables.day_index(pd.Series([lab["date"].max() + pd.Timedelta(days=1)]))


def test_real_data_window_counts_match_phase1(real_raw):
    lab = _labelled(real_raw)
    hist = add_historical_delay_rates(lab, ["carrier", "origin"], [1, 7])
    tables, idx, _ = _sequences(lab)
    for entity in ("carrier", "origin"):
        n = tables.counts[entity][0]
        for w in (1, 7):
            steps = idx["day"][:, None] + np.arange(-w, 0)[None, :]
            window = n[steps, idx[entity][:, None]].sum(1)
            np.testing.assert_array_equal(window, hist[f"hist_{entity}_{w}d_n"].to_numpy())
