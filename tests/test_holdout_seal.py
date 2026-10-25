"""The Nov-Dec test holdout stays sealed during development (docs/decision_log.md, D-002)."""

import json
import re

import pandas as pd
import pytest

from flight_delay.config import REPO_ROOT, load_config
from flight_delay.data.split import assert_no_holdout

# Files allowed to name the sealed full-year file. The final test evaluation script is added here
# when it is written, and is run once.
SEALED_READERS = {"scripts/seal_holdout.py", "scripts/evaluate.py"}


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


def test_only_allowed_files_reference_the_sealed_data():
    pattern = re.compile(r"raw_sealed|data/raw/flights\.csv|test\.parquet")
    offenders = []
    for folder in ("src", "scripts", "notebooks"):
        for path in (REPO_ROOT / folder).rglob("*"):
            if path.suffix not in {".py", ".ipynb"} or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in SEALED_READERS:
                continue
            text = path.read_text(errors="ignore")
            if path.suffix == ".ipynb":  # code only; saved outputs are not code that reads data
                cells = json.loads(text)["cells"]
                text = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
            if pattern.search(text):
                offenders.append(rel)
    assert not offenders, f"files referencing sealed holdout data: {offenders}"
