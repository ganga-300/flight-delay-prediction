"""Historical delay statistics that use strictly earlier dates only.

Decision time (``configs/config.yaml -> decision_time.mode = d_minus_1``): features for a flight
on date D are computed at the end of day D-1 (New York time). A past flight may contribute only
if its outcome was *known* before D, not just if it was scheduled before D.

Outcome-known time of a past flight = its actual arrival converted to New York time (destination
time zone, DST-aware, arrival date resolved; see ``timebase.py``) + ``REPORTING_BUFFER``. If the
arrival cannot be placed (unknown airport time zone), a conservative fallback is used:
departure + ``dep_delay`` + ``air_time`` + ``FALLBACK_BUFFER``, which exceeds the longest taxi time
observed (235 min). ``dep_delay``, ``air_time`` and ``arr_time`` are post-departure fields used
only to time when *earlier* flights' labels became available. They are never features of the
flight being predicted. A labelled flight with no departure record (only under the ``delayed``
policy) is treated as known at the end of the day after its scheduled date.

For each key (carrier, route, origin, destination) and window w, the features on date D
aggregate flights whose outcome became known on dates D-w ... D-1:

* ``hist_{key}_{w}_n``    — number of such flights
* ``hist_{key}_{w}_rate`` — delay rate shrunk towards the global past rate:
  (k + m * p0) / (n + m), where p0 is the all-past global delay rate known before D and m is
  the prior strength. With no history (unseen route, new carrier) the feature equals p0; on
  the very first date, when there is no past data at all, it is NaN (imputed later from
  training data by FeatureEncoder).

Assumption (not verifiable from this dataset): flight outcomes are available in near real
time, e.g. from an operational flight-status feed. Public BTS on-time data is released monthly
with a lag, so a system relying on BTS files alone could not compute the 1/7/28-day windows.
"""

from __future__ import annotations

import pandas as pd

from flight_delay.features.timebase import actual_arrival_ny

REPORTING_BUFFER = pd.Timedelta(minutes=30)
FALLBACK_BUFFER = pd.Timedelta(minutes=240)
TARGET = "is_delayed"


def outcome_known_date(df: pd.DataFrame) -> pd.Series:
    """New York calendar date on which each flight's arrival outcome is assumed to be known."""
    known = actual_arrival_ny(df) + REPORTING_BUFFER
    # Estimate the outcome-known time in UTC so flights from other
    # time zones are assigned to the correct New York calendar date.
    from flight_delay.features.timebase import localize_to_utc

    estimated_departure_local = (
        df["sched_dep_dt"]
        + pd.to_timedelta(df["dep_delay"], unit="m")
    )
    estimated_departure_utc = localize_to_utc(
        estimated_departure_local, df["origin"]
    )
    estimate_utc = (
        estimated_departure_utc
        + pd.to_timedelta(df["air_time"], unit="m")
        + FALLBACK_BUFFER
    )
    estimate_ny = (
        estimate_utc.dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    known = known.where(known.notna(), estimate_ny)
    known = known.where(
        known.notna(), df["date"] + pd.Timedelta(days=2)
    )  # no usable arrival or departure-time estimate
    return known.dt.normalize()


def _window_label(w: int | None) -> str:
    return "all" if w is None else f"{w}d"


def _past_window_sums(daily: pd.DataFrame, dates: pd.DatetimeIndex, w: int | None) -> pd.DataFrame:
    """daily: index = known date, columns = keys. Returns sums over [D-w, D-1] for each D."""
    daily = daily.reindex(dates, fill_value=0)
    if w is None:
        past = daily.cumsum().shift(1, fill_value=0)
    else:
        past = daily.rolling(w, min_periods=1).sum().shift(1, fill_value=0)
    return past


def add_historical_delay_rates(
    df: pd.DataFrame,
    keys: list[str],
    windows_days: list[int],
    include_all_past: bool = True,
    prior_strength: float = 20.0,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add past-only delay counts and smoothed rates per key and window.

    Parameters
    ----------
    df : rows to add features to (needs ``date`` and every column in ``keys``).
    history : labelled flights whose outcomes may be used (needs ``is_delayed``, ``date``,
        ``sched_dep_dt``, ``dep_delay``, ``air_time`` and the key columns). Defaults to ``df``.
        Rows with a missing label are ignored.
    """
    history = df if history is None else history
    hist = history[history[TARGET].notna()].copy()
    hist["_known"] = outcome_known_date(hist)
    hist["_y"] = hist[TARGET].astype(float)

    start = min(df["date"].min(), hist["date"].min())
    end = df["date"].max()
    dates = pd.date_range(start, end, freq="D")
    windows: list[int | None] = list(windows_days) + ([None] if include_all_past else [])

    out = df.copy()
    # Global prior: all-past delay rate known before D (same for every key).
    g_n = hist.groupby("_known")["_y"].size()
    g_k = hist.groupby("_known")["_y"].sum()
    p0_n = _past_window_sums(g_n.to_frame("x"), dates, None)["x"]
    p0_k = _past_window_sums(g_k.to_frame("x"), dates, None)["x"]
    p0 = (p0_k / p0_n).where(p0_n > 0)
    out["hist_global_all_rate"] = out["date"].map(p0).astype(float)
    out["hist_global_all_n"] = out["date"].map(p0_n).astype(float)

    for key in keys:
        n_daily = hist.pivot_table(
            index="_known", columns=key, values="_y", aggfunc="size", fill_value=0, observed=True
        )
        k_daily = hist.pivot_table(
            index="_known", columns=key, values="_y", aggfunc="sum", fill_value=0, observed=True
        )
        for w in windows:
            label = _window_label(w)
            n_past = _past_window_sums(n_daily, dates, w).stack().rename("n")
            k_past = _past_window_sums(k_daily, dates, w).stack().rename("k")
            past = pd.concat([n_past, k_past], axis=1)
            past.index.names = ["date", key]
            joined = out[["date", key]].join(past, on=["date", key])
            n = joined["n"].fillna(0.0)  # key never seen in the past -> 0 flights
            k = joined["k"].fillna(0.0)
            prior = out["hist_global_all_rate"]
            out[f"hist_{key}_{label}_n"] = n.astype(float)
            out[f"hist_{key}_{label}_rate"] = (k + prior_strength * prior) / (n + prior_strength)
    return out
