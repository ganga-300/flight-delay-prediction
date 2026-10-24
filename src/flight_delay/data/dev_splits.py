"""Train and validation data for model development (training, tuning, model selection).

This is the only way model code should read data. It returns the encoded feature matrices
written by ``scripts/build_features.py``. The test split cannot be requested: the Nov-Dec holdout
is sealed (``docs/decision_log.md``, D-002) and is read once, through ``data.holdout``, after the
model is frozen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from flight_delay.config import REPO_ROOT, load_config
from flight_delay.data.schema import TARGET_COLUMN
from flight_delay.data.split import assert_no_holdout
from flight_delay.features.build import CATEGORICAL, numeric_features
from flight_delay.features.leakage import assert_no_leakage

DEV_SPLITS = ("train", "valid")


def _processed_dir(cfg: dict, root: str | Path | None) -> Path:
    return Path(root or REPO_ROOT) / cfg["paths"]["processed"]


def feature_columns(cfg: dict | None = None, root: str | Path | None = None) -> dict[str, list[str]]:
    """Numeric and categorical model features, as written by the frozen feature build.

    Raises if the processed data was built with a different feature set than the pinned one,
    so a model can never be trained on a stale or edited feature list.
    """
    cfg = cfg or load_config()
    spec = json.loads((_processed_dir(cfg, root) / "feature_spec.json").read_text())
    numeric, categorical = list(spec["numeric"]), list(spec["categorical"])
    if numeric != numeric_features(cfg) or categorical != CATEGORICAL:
        raise ValueError(
            "data/processed/feature_spec.json does not match the frozen feature set. "
            "Rebuild with: python scripts/build_features.py"
        )
    assert_no_leakage(numeric + categorical)
    return {"numeric": numeric, "categorical": categorical}


def load_dev_split(
    name: str, cfg: dict | None = None, root: str | Path | None = None
) -> pd.DataFrame:
    """Load the encoded ``train`` or ``valid`` split. Asking for ``test`` raises."""
    if name not in DEV_SPLITS:
        raise ValueError(
            f"split {name!r} is not available for model development; only {DEV_SPLITS}. "
            "The test holdout is sealed and is evaluated once, after the model is frozen."
        )
    cfg = cfg or load_config()
    df = pd.read_parquet(_processed_dir(cfg, root) / f"{name}.parquet")
    start, end = (pd.Timestamp(d) for d in cfg["split"][name])
    dates = pd.to_datetime(df["date"])
    assert_no_holdout(dates, cfg["split"])
    if not dates.between(start, end).all():
        raise ValueError(f"{name}.parquet holds rows outside {start.date()}..{end.date()}")
    if TARGET_COLUMN not in df:
        raise ValueError(f"{name}.parquet has no target column {TARGET_COLUMN!r}")
    return df
