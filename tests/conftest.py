"""Shared fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flight_delay.config import REPO_ROOT, load_config

# Real-data tests read the development file (Jan-Oct). The Nov-Dec holdout stays sealed.
RAW_CSV = REPO_ROOT / load_config()["paths"]["raw"]


@pytest.fixture
def tiny_flights() -> pd.DataFrame:
    """Small hand-made frame spanning several dates/carriers/airports."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-01", "2013-01-02", "2013-01-03"]),
            "carrier": ["UA", "AA", "UA"],
            "origin": ["JFK", "LGA", "EWR"],
            "dest": ["LAX", "ORD", "LAX"],
            "sched_dep_time": [600, 1230, 1945],
            "arr_delay": [-5.0, 30.0, 12.0],
        }
    )


def make_raw(n_days: int = 40, per_day: int = 30, seed: int = 0) -> pd.DataFrame:
    """Deterministic synthetic table with the raw Kaggle schema, already passed through the
    same date parsing as ``load_raw``. Includes cancelled flights (missing actual times)."""
    rng = np.random.default_rng(seed)
    rows = []
    carriers, origins, dests = ["UA", "AA", "DL"], ["EWR", "JFK", "LGA"], ["ORD", "LAX", "ATL"]
    i = 0
    for d in pd.date_range("2013-01-01", periods=n_days, freq="D"):
        for _ in range(per_day):
            h, m = int(rng.integers(5, 23)), int(rng.choice([0, 15, 30, 45]))
            sched_dep = h * 100 + m
            block = int(rng.integers(90, 300))
            arr_min = (h * 60 + m + block) % 1440
            sched_arr = (arr_min // 60) * 100 + arr_min % 60
            cancelled = rng.random() < 0.03
            dep_delay = np.nan if cancelled else float(rng.integers(-10, 120))
            arr_delay = np.nan if cancelled else dep_delay + float(rng.integers(-20, 20))
            air = np.nan if cancelled else float(block - 20)
            rows.append(
                dict(
                    id=i,
                    year=d.year,
                    month=d.month,
                    day=d.day,
                    dep_time=np.nan if cancelled else float(sched_dep),
                    sched_dep_time=sched_dep,
                    dep_delay=dep_delay,
                    arr_time=np.nan if cancelled else float(sched_arr),
                    sched_arr_time=sched_arr,
                    arr_delay=arr_delay,
                    carrier=str(rng.choice(carriers)),
                    flight=int(rng.integers(1, 2000)),
                    tailnum=None if cancelled else "N1",
                    origin=str(rng.choice(origins)),
                    dest=str(rng.choice(dests)),
                    air_time=air,
                    distance=int(rng.integers(200, 2500)),
                    hour=h,
                    minute=m,
                    time_hour=d + pd.Timedelta(hours=h),
                    name="x",
                )
            )
            i += 1
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df["sched_dep_dt"] = df["date"] + pd.to_timedelta(df["hour"] * 60 + df["minute"], unit="m")
    return df


@pytest.fixture
def synthetic_raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def synthetic_config() -> dict:
    cfg = load_config()
    cfg["split"] = {
        "train": ["2013-01-01", "2013-01-25"],
        "valid": ["2013-01-26", "2013-02-02"],
        "test": ["2013-02-03", "2013-02-09"],
    }
    cfg["features"]["categorical_min_count"] = 5
    return cfg


@pytest.fixture(scope="session")
def real_raw() -> pd.DataFrame:
    if not Path(RAW_CSV).exists():
        pytest.skip("development data missing: run python scripts/seal_holdout.py")
    from flight_delay.data.load import load_dev_raw

    return load_dev_raw(load_config())
