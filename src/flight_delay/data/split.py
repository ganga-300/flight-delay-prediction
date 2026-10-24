"""Chronological train / validation / test split."""

from __future__ import annotations

import pandas as pd

SPLIT_ORDER = ("train", "valid", "test")


def assign_split(dates: pd.Series, ranges: dict[str, list[str] | tuple[str, str]]) -> pd.Series:
    """Label each date with the split whose inclusive date range contains it.

    Ranges must be ordered train < valid < test and must not overlap. Dates outside every
    range get ``<NA>``.
    """
    bounds = {k: (pd.Timestamp(v[0]), pd.Timestamp(v[1])) for k, v in ranges.items()}
    names = [s for s in SPLIT_ORDER if s in bounds] + [s for s in bounds if s not in SPLIT_ORDER]
    for name in names:
        start, end = bounds[name]
        if start > end:
            raise ValueError(f"split {name!r}: start {start.date()} is after end {end.date()}")
    for a, b in zip(names[:-1], names[1:], strict=True):
        if bounds[a][1] >= bounds[b][0]:
            raise ValueError(f"splits {a!r} and {b!r} overlap or are out of order")

    out = pd.Series(pd.NA, index=dates.index, dtype="string")
    for name in names:
        start, end = bounds[name]
        out[dates.between(start, end)] = name
    return out


def assert_no_holdout(dates: pd.Series, ranges: dict[str, list[str] | tuple[str, str]]) -> None:
    """Raise if any date falls after the validation period, i.e. in the sealed test holdout.

    Development code (EDA, features, training, model selection) must never see test-period rows.
    """
    valid_end = pd.Timestamp(ranges["valid"][1])
    late = dates > valid_end
    if late.any():
        raise ValueError(
            f"{int(late.sum()):,} rows are dated after {valid_end.date()} (sealed test holdout). "
            "Development code must read paths.raw (Jan-Oct), not the sealed full-year file."
        )


def chronological_split(
    df: pd.DataFrame, date_col: str, ranges: dict[str, list[str] | tuple[str, str]]
) -> dict[str, pd.DataFrame]:
    """Split by inclusive date ranges, e.g. {"train": ("2013-01-01", "2013-08-31"), ...}."""
    labels = assign_split(df[date_col], ranges)
    return {name: df.loc[labels == name].copy() for name in ranges}
