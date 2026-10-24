"""Load the raw Kaggle *Flights* (NYC departures, 2013) CSV with an explicit schema.

The raw file has 21 columns. Times are integers in local ``HHMM`` format; the actual
times / delays are missing for cancelled flights, so they are read as nullable floats.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Column -> pandas dtype used when reading the raw CSV.
RAW_DTYPES: dict[str, str] = {
    "id": "int64",
    "year": "int16",
    "month": "int8",
    "day": "int8",
    "dep_time": "float64",  # actual, HHMM local; NaN if cancelled
    "sched_dep_time": "int16",  # HHMM local
    "dep_delay": "float64",  # minutes; NaN if cancelled
    "arr_time": "float64",  # actual, HHMM local at destination
    "sched_arr_time": "int16",  # HHMM local at destination
    "arr_delay": "float64",  # minutes; NaN if cancelled / no arrival record
    "carrier": "string",
    "flight": "int32",
    "tailnum": "string",
    "origin": "string",
    "dest": "string",
    "air_time": "float64",  # minutes airborne
    "distance": "int32",  # miles
    "hour": "int8",  # scheduled departure hour
    "minute": "int8",  # scheduled departure minute
    "time_hour": "string",  # scheduled departure hour as text, parsed below
    "name": "string",  # carrier full name
}

RAW_COLUMNS: list[str] = list(RAW_DTYPES)


def hhmm_to_minutes(hhmm: pd.Series) -> pd.Series:
    """Convert local ``HHMM`` integers (e.g. 1745) to minutes after midnight.

    ``2400`` (used in the data for exactly midnight) maps to 1440. Missing values stay NaN.
    """
    hhmm = pd.to_numeric(hhmm, errors="coerce")
    return (hhmm // 100) * 60 + (hhmm % 100)


def load_raw(path: str | Path, parse_dates: bool = True) -> pd.DataFrame:
    """Read the raw CSV with explicit dtypes.

    If ``parse_dates`` is True, also adds:

    * ``date`` — flight date (``datetime64``), derived from year/month/day.
    * ``sched_dep_dt`` — scheduled departure as a naive local datetime (origin airport).
    * ``time_hour`` — parsed to ``datetime64``.

    The function never drops or modifies raw values; cleaning happens in ``clean.py``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place the Kaggle 'flights.csv' in data/raw/ (see data/README.md)."
        )
    df = pd.read_csv(path, dtype=RAW_DTYPES)

    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw file is missing expected columns: {sorted(missing)}")

    if parse_dates:
        df["date"] = pd.to_datetime(df[["year", "month", "day"]].astype("int64"))
        df["sched_dep_dt"] = df["date"] + pd.to_timedelta(
            hhmm_to_minutes(df["sched_dep_time"]).astype(np.int64), unit="m"
        )
        df["time_hour"] = pd.to_datetime(df["time_hour"])
    return df


def load_dev_raw(cfg: dict, root: str | Path | None = None) -> pd.DataFrame:
    """Load the development raw file (``paths.raw``) and check it holds no holdout rows.

    Use this everywhere except the final test evaluation. It raises if any row is dated after the
    validation period, so the sealed Nov-Dec data cannot reach development code by mistake.
    """
    from flight_delay.config import REPO_ROOT
    from flight_delay.data.split import assert_no_holdout

    path = Path(root or REPO_ROOT) / cfg["paths"]["raw"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Create it with: python scripts/seal_holdout.py")
    df = load_raw(path)
    assert_no_holdout(df["date"], cfg["split"])
    return df
