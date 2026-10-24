"""Experiment plan and run records for model training (``docs/decision_log.md``, D-201, D-108).

Training code calls ``write_run_record`` after each run. It refuses a run whose hyperparameters
are not in the approved plan (``configs/experiment_plan.yaml``), and any metric that is not a
train or validation metric. Each record stores the plan's hash, the commit and the feature set,
so ``tests/test_experiments.py`` can check that every committed run followed the plan.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from flight_delay.config import REPO_ROOT

PLAN_FILE = "configs/experiment_plan.yaml"
RUNS_DIR = "experiments/runs"
METRIC_PREFIXES = ("train_", "valid_")
PLAN_STATUSES = ("draft", "approved")


def load_plan(root: str | Path | None = None) -> dict:
    plan = yaml.safe_load((Path(root or REPO_ROOT) / PLAN_FILE).read_text())
    if plan.get("status") not in PLAN_STATUSES:
        raise ValueError(f"plan status must be one of {PLAN_STATUSES}, got {plan.get('status')!r}")
    if not str(plan.get("selection_metric", "")).startswith("valid_"):
        raise ValueError("plan selection_metric must be a validation metric (valid_...)")
    if not isinstance(plan.get("models"), dict):
        raise ValueError("plan must list models as a mapping")
    return plan


def plan_sha256(root: str | Path | None = None) -> str:
    return hashlib.sha256((Path(root or REPO_ROOT) / PLAN_FILE).read_bytes()).hexdigest()


def check_params(model: str, params: dict, plan: dict) -> None:
    """Raise unless ``params`` is one point of the approved grid for ``model``."""
    if plan["status"] != "approved":
        raise PermissionError("the experiment plan is still a draft: record and approve D-201 first")
    if model not in plan["models"]:
        raise ValueError(f"model {model!r} is not in the experiment plan")
    grid = plan["models"][model] or {}
    if set(params) != set(grid):
        raise ValueError(
            f"{model}: parameters {sorted(params)} must match the plan's {sorted(grid)} exactly"
        )
    for key, value in params.items():
        if value not in grid[key]:
            raise ValueError(f"{model}: {key}={value!r} is not in the plan ({grid[key]})")


def check_metrics(metrics: dict) -> None:
    """Raise unless every metric is a train or validation metric."""
    bad = [k for k in metrics if not k.startswith(METRIC_PREFIXES)]
    if bad:
        raise ValueError(f"only train_/valid_ metrics may be recorded, got {bad}")


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def write_run_record(
    model: str,
    params: dict,
    metrics: dict,
    features: list[str],
    seed: int,
    root: str | Path | None = None,
    notes: str = "",
) -> Path:
    """Validate a finished run against the plan and save its record under ``experiments/runs``."""
    root = Path(root or REPO_ROOT)
    plan = load_plan(root)
    check_params(model, params, plan)
    check_metrics(metrics)
    now = datetime.now(timezone.utc)
    record = {
        "model": model,
        "params": params,
        "metrics": metrics,
        "seed": seed,
        "features_sha256": hashlib.sha256(json.dumps(features).encode()).hexdigest(),
        "n_features": len(features),
        "plan_sha256": plan_sha256(root),
        "commit": _git(root, "rev-parse", "HEAD"),
        # Uncommitted code changes; earlier, not-yet-committed run records do not count.
        "dirty": bool(_git(root, "status", "--porcelain", "--", ".", f":(exclude){RUNS_DIR}")),
        "created_utc": now.isoformat(timespec="seconds"),
        "notes": notes,
    }
    out_dir = root / RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{now:%Y%m%dT%H%M%S}_{model}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
