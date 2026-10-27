"""Assemble the leakage-safe feature matrix.

Pipeline (``build_feature_table`` + ``encode_splits``):

1. Congestion counts on the FULL raw table (includes flights later dropped for missing labels).
2. Target (``arr_delay >= threshold``) and missing-label policy (``clean.build_labelled_frame``).
3. Calendar, schedule and route features: pure functions of the published schedule.
4. Past-only historical delay statistics (``historical.py``).
5. Column selection: only names listed in ``FEATURE_GROUPS`` are kept, then
   ``assert_no_leakage`` runs.
6. Chronological split (config dates) -> ``FeatureEncoder`` fitted on train only -> transform.
"""

from __future__ import annotations

import pandas as pd

from flight_delay.data.clean import build_labelled_frame
from flight_delay.data.schema import TARGET_COLUMN
from flight_delay.data.split import assign_split
from flight_delay.features.calendar import add_calendar_features
from flight_delay.features.congestion import CONGESTION_SLOTS, add_congestion_features
from flight_delay.features.encoding import FeatureEncoder
from flight_delay.features.historical import add_historical_delay_rates
from flight_delay.features.leakage import POST_DEPARTURE_COLUMNS, assert_no_leakage
from flight_delay.features.route import add_route_features
from flight_delay.features.schedule import add_schedule_features

HISTORY_KEYS = ["carrier", "route", "origin", "dest"]
META_COLUMNS = ["id", "date", "split"]

CATEGORICAL = [
    "carrier",
    "origin",
    "dest",
    "route",
    "dep_bucket",
    "carrier_origin",
    "carrier_dest",
    "carrier_route",
    "origin_dep_bucket",
    "carrier_dep_bucket",
]
CALENDAR = [
    "sched_dep_hour_frac",
    "dow",
    # Month / day-of-year encodings (raw and cyclic) are NOT used. Training covers Jan-Aug only,
    # so every validation/test month is unseen and these features can only extrapolate.
    # Seasonality is carried by past-only history windows instead (notebook 03, audit §A3).
    "is_weekend",
    "is_holiday",
    "days_from_holiday",
    "abs_days_to_holiday",
    "dep_hour_sin",
    "dep_hour_cos",
    "dow_sin",
    "dow_cos",
]

SCHEDULE = [
    "sched_dep_min",
    "sched_arr_min",
    "sched_block_min",
    "overnight_arrival",
    "distance",
    "log_distance",
]
CONGESTION = [*CONGESTION_SLOTS, "sched_dep_origin_hour_share"]



def history_columns(windows_days: list[int], include_all_past: bool = True) -> list[str]:
    """Names of historical features used by the model.

    Expanding ("all") windows contribute only their smoothed *rate*. Their cumulative counts
    (``hist_*_all_n``) are computed but not used as features: they grow monotonically with time,
    so under a chronological split every test value is larger than any training value. They act
    as a time index rather than a delay signal (see notebooks/03_feature_engineering.ipynb).
    """
    cols = ["hist_global_all_rate"]
    for key in HISTORY_KEYS:
        for w in windows_days:
            cols += [f"hist_{key}_{w}d_rate", f"hist_{key}_{w}d_n"]
        if include_all_past:
            cols.append(f"hist_{key}_all_rate")
    return cols


def feature_groups(config: dict) -> dict[str, list[str]]:
    fcfg = config.get("features", {})
    hist = history_columns(
        fcfg.get("history_windows_days", [1, 7, 28]), fcfg.get("history_include_all_past", True)
    )
    return {
        "calendar": CALENDAR, "schedule_route": SCHEDULE,
        "categorical": CATEGORICAL, "congestion": CONGESTION,
        "historical": hist,
    }


def numeric_features(config: dict) -> list[str]:
    g = feature_groups(config)
    return g["calendar"] + g["schedule_route"] + g["congestion"] + g["historical"]



def build_feature_table(raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Raw table (from ``load_raw``) -> labelled rows with all features, split label, target.

    Every value for a flight on date D depends only on the published schedule and on outcomes
    known before D. No learned transformation is applied here.
    """
    tcfg, fcfg = config["target"], config.get("features", {})
    df = add_congestion_features(raw)  # step 1: full raw schedule
    df = build_labelled_frame(df, tcfg["threshold_minutes"], tcfg["cancelled_policy"])  # step 2
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int8")
    df = add_calendar_features(df)
    df = add_schedule_features(df)
    df = add_route_features(df)
    df = add_historical_delay_rates(  # step 4 (history = labelled rows only)
        df,
        keys=HISTORY_KEYS,
        windows_days=fcfg.get("history_windows_days", [1, 7, 28]),
        include_all_past=fcfg.get("history_include_all_past", True),
        prior_strength=fcfg.get("history_prior_strength", 20.0),
    )
    df["split"] = assign_split(df["date"], config["split"])

    feats = numeric_features(config) + CATEGORICAL
    table = df[META_COLUMNS + feats + [TARGET_COLUMN]].copy()  # step 5: explicit allow-list
    assert_no_leakage(feats)
    forbidden = POST_DEPARTURE_COLUMNS | {TARGET_COLUMN, "flight_status", "tailnum"}
    assert not forbidden.intersection(feats), forbidden.intersection(feats)
    return table.sort_values(["date", "id"]).reset_index(drop=True)


def encode_splits(
    table: pd.DataFrame, config: dict
) -> tuple[dict[str, pd.DataFrame], FeatureEncoder]:
    """Fit ``FeatureEncoder`` on the train split only and transform every split."""
    enc = FeatureEncoder(
        CATEGORICAL,
        numeric_features(config),
        min_count=config.get("features", {}).get("categorical_min_count", 50),
    )
    enc.fit(table[table["split"] == "train"])
    splits = {
        name: enc.transform(table[table["split"] == name].reset_index(drop=True))
        for name in ("train", "valid", "test")
    }
    return splits, enc



def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Backward-compatible entry point: raw table -> unencoded feature table."""
    return build_feature_table(df, config)
