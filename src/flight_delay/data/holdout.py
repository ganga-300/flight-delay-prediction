"""One-shot access to the sealed Nov-Dec test holdout (``docs/decision_log.md``, D-002, D-108).

``open_sealed_holdout`` is the only code allowed to read the full-year file. It refuses unless:

* the current commit carries the ``model-frozen`` tag, so the model was fixed before any test
  result existed;
* the working tree is clean, so the evaluated code is exactly the tagged commit;
* no test results have been written yet, so the test set is evaluated once.

After the evaluation, commit ``reports/results/test_metrics.json``. Its presence keeps the gate
closed for good.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pandas as pd

from flight_delay.config import REPO_ROOT, load_config
from flight_delay.data.load import load_raw

SEALED_SHA256 = "8bdc6abbf07ac0cbb1036bc14a787cab3eea616be46c31899d2d8048963b08aa"
FROZEN_TAG = "model-frozen"
RESULTS_FILE = "reports/results/test_metrics.json"


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def holdout_gate_problems(root: str | Path | None = None) -> list[str]:
    """Reasons the holdout must stay closed. An empty list means it may be opened."""
    root = Path(root or REPO_ROOT)
    problems = []
    if FROZEN_TAG not in _git(root, "tag", "--points-at", "HEAD").split():
        problems.append(f"the current commit is not tagged {FROZEN_TAG!r}")
    if _git(root, "status", "--porcelain"):
        problems.append("the working tree has uncommitted changes")
    if (root / RESULTS_FILE).exists():
        problems.append(f"{RESULTS_FILE} already exists: the test set has been evaluated once")
    return problems


def open_sealed_holdout(cfg: dict | None = None, root: str | Path | None = None) -> pd.DataFrame:
    """Return the full-year raw table for the final evaluation, if the gate allows it."""
    root = Path(root or REPO_ROOT)
    problems = holdout_gate_problems(root)
    if problems:
        raise PermissionError("The sealed test holdout stays closed:\n- " + "\n- ".join(problems))
    cfg = cfg or load_config()
    path = root / cfg["paths"]["raw_sealed"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != SEALED_SHA256:
        raise ValueError(f"{path} does not match the sealed SHA-256 recorded in D-002")
    return load_raw(path)
