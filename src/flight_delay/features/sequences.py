"""Ordered daily delay sequences for the Member 2 GRU (docs/decision_log.md, D-303).

For a flight on day D the sequence covers the last L days, D-L ... D-1, oldest first. A past
flight counts on the New York date on which its outcome became known (``outcome_known_date``,
the same rule as the Phase 1 history features), so a flight from D-1 that lands after 23:30
is not part of D's sequence.

Ten values per day (``SEQUENCE_CHANNELS``):

* delay rate of the whole system, the flight's origin, carrier and destination, each shrunk
  towards the global past rate: (k + m * p_t) / (n + m), where p_t is the global rate of all
  outcomes known up to and including day t, and m is the prior strength;
* log(1 + number of known outcomes) for system, origin and carrier;
* a has-data flag for the carrier (1 if it had any known outcome that day);
* the weekday of the day as sine/cosine.

Daily values are stored once per (day, entity) in ``SequenceTables``. A flight only stores
indices into those tables, and sequences are gathered on demand. Days before the first date
of the data are padding: rates are NaN, counts and flags 0. ``SequenceScaler`` (fitted on
training flights only) standardises each channel and maps NaN to 0, the training mean.

Only past delayed / not-delayed labels are used: no delay minutes, no cancellations, and
nothing from day D or later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from flight_delay.features.historical import outcome_known_date

TARGET = "is_delayed"
ENTITIES = ("origin", "carrier", "dest")
SEQUENCE_CHANNELS = [
    "system_rate",
    "system_log_n",
    "origin_rate",
    "origin_log_n",
    "carrier_rate",
    "carrier_log_n",
    "carrier_has_data",
    "dest_rate",
    "dow_sin",
    "dow_cos",
]
UNKNOWN = "__UNKNOWN__"


@dataclass
class SequenceTables:
    """Per-day values, stored once per (day, entity).

    ``dates[i]`` is the calendar day of row ``i`` in every array. The first ``pad`` rows are
    padding days before the data starts.
    """

    dates: pd.DatetimeIndex
    pad: int
    levels: dict[str, list[str]]  # entity -> vocabulary; the last level is UNKNOWN
    counts: dict[str, tuple[np.ndarray, np.ndarray]]  # entity or "system" -> (n, k), [T, E]
    system: np.ndarray  # [T, 2]  rate, log_n
    origin: np.ndarray  # [T, E_origin, 2]  rate, log_n
    carrier: np.ndarray  # [T, E_carrier, 3]  rate, log_n, has_data
    dest: np.ndarray  # [T, E_dest, 1]  rate
    calendar: np.ndarray  # [T, 2]  dow_sin, dow_cos

    def day_index(self, dates: pd.Series) -> np.ndarray:
        pos = self.dates.get_indexer(pd.DatetimeIndex(dates))
        if (pos < 0).any():
            raise ValueError("flight dates outside the sequence tables")
        if (pos < self.pad).any():
            raise ValueError("flight dates fall in the padding days")
        return pos.astype(np.int64)

    def entity_index(self, entity: str, values: pd.Series) -> np.ndarray:
        vocab = pd.Index(self.levels[entity])
        pos = vocab.get_indexer(values.astype(str))
        return np.where(pos < 0, len(vocab) - 1, pos).astype(np.int64)

    def flight_indices(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Indices that locate each flight's sequence (needs date, origin, carrier, dest)."""
        out = {"day": self.day_index(frame["date"])}
        for entity in ENTITIES:
            out[entity] = self.entity_index(entity, frame[entity])
        return out

    def daily_arrays(self) -> list[np.ndarray]:
        return [self.system, self.origin, self.carrier, self.dest, self.calendar]


def build_sequence_tables(
    history: pd.DataFrame,
    end_date: pd.Timestamp,
    max_len: int = 28,
    prior_strength: float = 20.0,
) -> SequenceTables:
    """Daily sequence values from labelled past flights.

    Parameters
    ----------
    history : labelled flights whose outcomes may be used (needs ``is_delayed``, ``date``,
        ``sched_dep_dt``, ``dep_delay``, ``air_time``, ``arr_time``, origin, carrier, dest).
        Rows with a missing label are ignored. Post-departure fields only time the outcomes.
    end_date : last flight date that will need a sequence.
    max_len : longest sequence that will be gathered (number of padding days).
    """
    hist = history[history[TARGET].notna()].copy()
    hist["_known"] = outcome_known_date(hist)
    hist["_y"] = hist[TARGET].astype(float)

    start = hist["date"].min()
    end = pd.Timestamp(end_date)
    dates = pd.date_range(start - pd.Timedelta(days=max_len), end, freq="D")
    real = dates >= start

    def daily(key: str | None, vocab: list[str] | None) -> tuple[np.ndarray, np.ndarray]:
        if key is None:
            n = hist.groupby("_known")["_y"].size().reindex(dates, fill_value=0)
            k = hist.groupby("_known")["_y"].sum().reindex(dates, fill_value=0)
            return n.to_numpy(float)[:, None], k.to_numpy(float)[:, None]
        n = hist.pivot_table(
            index="_known", columns=key, values="_y", aggfunc="size", fill_value=0, observed=True
        )
        k = hist.pivot_table(
            index="_known", columns=key, values="_y", aggfunc="sum", fill_value=0, observed=True
        )
        n = n.reindex(index=dates, columns=vocab, fill_value=0)
        k = k.reindex(index=dates, columns=vocab, fill_value=0)
        return n.to_numpy(float), k.to_numpy(float)

    levels = {e: sorted(hist[e].astype(str).unique()) + [UNKNOWN] for e in ENTITIES}
    counts = {"system": daily(None, None)}
    for entity in ENTITIES:
        counts[entity] = daily(entity, levels[entity])

    # Global rate of all outcomes known up to and including day t (NaN before any outcome).
    sys_n, sys_k = counts["system"]
    cum_n, cum_k = np.cumsum(sys_n[:, 0]), np.cumsum(sys_k[:, 0])
    prior = np.where(cum_n > 0, cum_k / np.maximum(cum_n, 1), np.nan)
    prior[~real] = np.nan

    def rate(entity: str) -> np.ndarray:
        n, k = counts[entity]
        r = (k + prior_strength * prior[:, None]) / (n + prior_strength)
        r[~real] = np.nan
        return r

    def log_n(entity: str) -> np.ndarray:
        return np.log1p(counts[entity][0])

    dow = dates.day_of_week.to_numpy()
    calendar = np.stack([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)], axis=1)
    return SequenceTables(
        dates=dates,
        pad=int((~real).sum()),
        levels=levels,
        counts=counts,
        system=np.stack([rate("system")[:, 0], log_n("system")[:, 0]], axis=1),
        origin=np.stack([rate("origin"), log_n("origin")], axis=2),
        carrier=np.stack(
            [rate("carrier"), log_n("carrier"), (counts["carrier"][0] > 0).astype(float)], axis=2
        ),
        dest=rate("dest")[:, :, None],
        calendar=calendar,
    )


def gather_sequences(
    arrays: list[np.ndarray], idx: dict[str, np.ndarray], length: int
) -> np.ndarray:
    """[B, length, 10] sequences for days D-length ... D-1 (works on raw or scaled arrays)."""
    system, origin, carrier, dest, calendar = arrays
    steps = idx["day"][:, None] + np.arange(-length, 0)[None, :]
    parts = [
        system[steps],
        origin[steps, idx["origin"][:, None]],
        carrier[steps, idx["carrier"][:, None]],
        dest[steps, idx["dest"][:, None]],
        calendar[steps],
    ]
    return np.concatenate(parts, axis=2).astype(np.float32)


@dataclass
class SequenceScaler:
    """Per-channel standardisation fitted on training flights; NaN becomes 0 (the mean)."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(
        cls, tables: SequenceTables, idx: dict[str, np.ndarray], length: int, chunk: int = 20000
    ) -> SequenceScaler:
        c = len(SEQUENCE_CHANNELS)
        total, total_sq, count = np.zeros(c), np.zeros(c), np.zeros(c)
        arrays = tables.daily_arrays()
        for start in range(0, len(idx["day"]), chunk):
            part = {key: val[start : start + chunk] for key, val in idx.items()}
            x = gather_sequences(arrays, part, length).reshape(-1, c).astype(np.float64)
            ok = ~np.isnan(x)
            total += np.where(ok, x, 0).sum(0)
            total_sq += np.where(ok, x * x, 0).sum(0)
            count += ok.sum(0)
        mean = total / np.maximum(count, 1)
        std = np.sqrt(np.maximum(total_sq / np.maximum(count, 1) - mean**2, 0))
        return cls(mean=mean, std=np.where(std > 1e-8, std, 1.0))

    def transform_tables(self, tables: SequenceTables) -> list[np.ndarray]:
        """Scaled copies of the daily arrays, so gathered sequences are already standardised."""
        bounds = np.cumsum([0, 2, 2, 3, 1, 2])  # channel ranges of system/origin/carrier/dest/cal
        out = []
        for i, arr in enumerate(tables.daily_arrays()):
            lo, hi = bounds[i], bounds[i + 1]
            scaled = (arr - self.mean[lo:hi]) / self.std[lo:hi]
            out.append(np.nan_to_num(scaled, nan=0.0).astype(np.float32))
        return out
