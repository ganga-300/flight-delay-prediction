"""Training follows the pre-registered experiment plan (docs/decision_log.md, D-201, D-108)."""

import json
import subprocess

import pytest

from flight_delay.config import REPO_ROOT
from flight_delay.data.dev_splits import feature_columns
from flight_delay.experiments import (
    RUNS_DIR,
    check_metrics,
    check_params,
    load_plan,
    plan_sha256,
)

APPROVED = {
    "status": "approved",
    "selection_metric": "valid_pr_auc",
    "models": {"xgboost": {"max_depth": [3, 5], "learning_rate": [0.1]}},
}


def test_plan_file_is_valid():
    plan = load_plan()
    assert plan["selection_metric"].startswith("valid_")


def test_draft_plan_refuses_every_run():
    with pytest.raises(PermissionError, match="draft"):
        check_params("xgboost", {"max_depth": 3, "learning_rate": 0.1}, {**APPROVED, "status": "draft"})


def test_runs_must_be_points_of_the_plan_grid():
    check_params("xgboost", {"max_depth": 5, "learning_rate": 0.1}, APPROVED)
    with pytest.raises(ValueError, match="not in the plan"):
        check_params("xgboost", {"max_depth": 9, "learning_rate": 0.1}, APPROVED)
    with pytest.raises(ValueError, match="must match"):
        check_params("xgboost", {"max_depth": 3}, APPROVED)
    with pytest.raises(ValueError, match="must match"):
        check_params("xgboost", {"max_depth": 3, "learning_rate": 0.1, "subsample": 0.8}, APPROVED)
    with pytest.raises(ValueError, match="not in the experiment plan"):
        check_params("random_forest", {}, APPROVED)


def test_only_train_and_validation_metrics_may_be_recorded():
    check_metrics({"train_pr_auc": 0.4, "valid_pr_auc": 0.3})
    for bad in ("test_pr_auc", "pr_auc", "holdout_roc_auc"):
        with pytest.raises(ValueError, match="train_/valid_"):
            check_metrics({bad: 0.3})


def _records():
    return sorted((REPO_ROOT / RUNS_DIR).glob("*.json"))


@pytest.mark.parametrize("path", _records(), ids=lambda p: p.name)
def test_committed_run_records_follow_the_plan(path):
    rec = json.loads(path.read_text())
    plan = load_plan()
    assert rec["plan_sha256"] == plan_sha256(), "the plan changed after this run was recorded"
    check_params(rec["model"], rec["params"], plan)
    check_metrics(rec["metrics"])
    assert not rec["dirty"], "run was made on uncommitted code"
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", rec["commit"], "HEAD"], cwd=REPO_ROOT
    )
    assert is_ancestor.returncode == 0, "run was made on a commit that is not in this history"
    cols = feature_columns()
    assert rec["n_features"] == len(cols["numeric"]) + len(cols["categorical"])
