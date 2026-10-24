"""The Nov-Dec test holdout stays sealed during development (docs/decision_log.md, D-002)."""

import json
import re

import pandas as pd
import pytest

from flight_delay.config import REPO_ROOT, load_config
from flight_delay.data.split import assert_no_holdout

# Code scans over src/, scripts/ and notebook code cells. Each pattern lists the only files allowed
# to contain it. Every allowed file is owner-reviewed (.github/CODEOWNERS); see D-108.
SCANS = {
    "sealed full-year data": (
        re.compile(r"raw_sealed|data/raw/flights\.csv|nycflights13|test\.parquet"),
        {"scripts/seal_holdout.py", "src/flight_delay/data/holdout.py"},
    ),
    "test-split selection": (
        re.compile(
            r"""[=!]=\s*['"]test['"]|\[\s*['"]test['"]\s*\]|2013-1[12]"""
            r"""|month\W{0,4}\s*(?:>=?\s*1[01]|==\s*1[12])|open_sealed_holdout"""
        ),
        {"src/flight_delay/data/holdout.py", "scripts/evaluate.py"},
    ),
    "direct file reading": (
        re.compile(r"\bread_(?:csv|parquet|table|excel|feather|pickle)\b"),
        {"src/flight_delay/data/load.py", "src/flight_delay/data/dev_splits.py"},
    ),
}


def _code_files():
    for folder in ("src", "scripts", "notebooks"):
        for path in sorted((REPO_ROOT / folder).rglob("*")):
            if path.suffix not in {".py", ".ipynb"} or "__pycache__" in path.parts:
                continue
            text = path.read_text(errors="ignore")
            if path.suffix == ".ipynb":  # code only; saved outputs are not code that reads data
                cells = json.loads(text)["cells"]
                text = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
            yield path.relative_to(REPO_ROOT).as_posix(), text


def test_development_config_does_not_read_the_sealed_file():
    paths = load_config()["paths"]
    assert paths["raw"] != paths["raw_sealed"]
    assert "raw" in paths and "raw_sealed" in paths


def test_development_data_holds_no_holdout_rows(real_raw):
    cfg = load_config()
    assert real_raw["date"].max() <= pd.Timestamp(cfg["split"]["valid"][1])


def test_guard_rejects_holdout_dates():
    cfg = load_config()
    ok = pd.Series(pd.to_datetime(["2013-01-01", "2013-10-31"]))
    assert_no_holdout(ok, cfg["split"])
    with pytest.raises(ValueError, match="sealed test holdout"):
        assert_no_holdout(pd.Series(pd.to_datetime(["2013-11-01"])), cfg["split"])


@pytest.mark.parametrize("scan", SCANS)
def test_only_allowed_files_touch_the_holdout(scan):
    pattern, allowed = SCANS[scan]
    offenders = sorted(
        f"{rel}: {pattern.search(text).group()!r}"
        for rel, text in _code_files()
        if rel not in allowed and pattern.search(text)
    )
    assert not offenders, f"{scan} outside the allowed files: {offenders}"
