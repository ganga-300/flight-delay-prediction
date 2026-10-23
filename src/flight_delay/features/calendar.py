"""Calendar and cyclic time features (all derived from the scheduled departure)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

HOLIDAY_WINDOW_DAYS = 14  # distances beyond this are clipped


def add_cyclic_encoding(
    df: pd.DataFrame, col: str, period: float, prefix: str | None = None
) -> pd.DataFrame:
    """Add ``{prefix}_sin`` and ``{prefix}_cos`` = sin/cos(2*pi*x/period).

    Values that differ by a whole period map to the same point, so 23:00 and 00:00 end up
    close together, as do December and January.
    """
    prefix = prefix or col
    angle = 2 * np.pi * df[col].astype(float) / period
    out = df.copy()
    out[f"{prefix}_sin"] = np.sin(angle)
    out[f"{prefix}_cos"] = np.cos(angle)
    return out


def us_federal_holidays(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """US federal holidays (rule-based, so deterministic) in [start, end]."""
    return USFederalHolidayCalendar().holidays(start=start, end=end)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Day of week, month, day of year, weekend, US federal holiday proximity, cyclic encodings.

    Requires ``date`` and ``sched_dep_dt`` (from ``load_raw``). Uses only the calendar and the
    published schedule, so every value is known before departure.
    """
    out = df.copy()
    date = out["date"]
    out["dow"] = date.dt.day_of_week.astype("int8")  # 0 = Monday
    out["month"] = date.dt.month.astype("int8")
    out["day_of_year"] = date.dt.day_of_year.astype("int16")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")

    # Pad the range so holidays just outside the data (e.g. 1 Jan of the next year) count.
    pad = pd.Timedelta(days=HOLIDAY_WINDOW_DAYS + 31)
    hol = us_federal_holidays(date.min() - pad, date.max() + pad).values.astype("datetime64[D]")
    d = date.values.astype("datetime64[D]")
    diff = (d[:, None] - hol[None, :]).astype(int)  # days since each holiday (negative = before)
    nearest = diff[np.arange(len(d)), np.abs(diff).argmin(axis=1)]
    out["is_holiday"] = (nearest == 0).astype("int8")
    out["days_from_holiday"] = np.clip(nearest, -HOLIDAY_WINDOW_DAYS, HOLIDAY_WINDOW_DAYS).astype(
        "int8"
    )
    out["abs_days_to_holiday"] = np.abs(out["days_from_holiday"]).astype("int8")

    dep_hour = (out["sched_dep_dt"] - date).dt.total_seconds() / 3600
    out["sched_dep_hour_frac"] = dep_hour
    out = add_cyclic_encoding(out, "sched_dep_hour_frac", 24, prefix="dep_hour")
    out = add_cyclic_encoding(out, "dow", 7)
    out = add_cyclic_encoding(out, "month", 12)  # Jan=1 ... Dec=12; Dec and Jan are adjacent
    out = add_cyclic_encoding(out, "day_of_year", 365.25, prefix="doy")
    return out
