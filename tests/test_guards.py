"""Guards that keep model development on train/validation data only (docs/decision_log.md, D-108)."""

import json
import subprocess

import pytest
import yaml

from flight_delay.config import load_config
from flight_delay.data.dev_splits import DEV_SPLITS, feature_columns, load_dev_split
from flight_delay.data.holdout import FROZEN_TAG, RESULTS_FILE, open_sealed_holdout
from flight_delay.experiments import PLAN_FILE, RUNS_DIR, write_run_record

# Frozen by D-002, D-010 to D-014 and D-102 to D-105. Changing any of these is a new, logged
# decision: update the log and this dictionary in the same owner-reviewed pull request.
LOCKED_CONFIG = {
    "seed": 42,
    "target": {"column": "arr_delay", "threshold_minutes": 15, "cancelled_policy": "exclude"},
    "decision_time": {"mode": "d_minus_1"},
    "split": {
        "train": ["2013-01-01", "2013-08-31"],
        "valid": ["2013-09-01", "2013-10-31"],
        "test": ["2013-11-01", "2013-12-31"],
    },
    "features": {
        "history_windows_days": [1, 7, 28],
        "history_include_all_past": True,
        "history_prior_strength": 20,
        "categorical_min_count": 50,
        "cyclic": ["hour", "day_of_week"],
    },
    "paths": {
        "raw": "data/interim/flights_dev.csv",
        "raw_sealed": "data/raw/flights.csv",
        "processed": "data/processed",
    },
}


def test_frozen_config_sections_are_unchanged():
    cfg = load_config()
    current = {k: cfg[k] for k in LOCKED_CONFIG if k != "paths"}
    current["paths"] = {k: cfg["paths"][k] for k in LOCKED_CONFIG["paths"]}
    assert current == LOCKED_CONFIG


# ---------------------------------------------------------------- train/validation data access
def test_model_code_can_load_only_train_and_validation():
    assert DEV_SPLITS == ("train", "valid")
    for name in ("test", "holdout", "all"):
        with pytest.raises(ValueError, match="not available for model development"):
            load_dev_split(name)


def test_dev_splits_match_the_frozen_feature_set():
    cols = feature_columns()
    feats = cols["numeric"] + cols["categorical"]
    assert len(feats) == 59
    for name in DEV_SPLITS:
        assert set(feats) <= set(load_dev_split(name).columns)


# ---------------------------------------------------------------- throwaway git repository
def _git(root, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=root, check=True, capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "full.csv").write_text("not the sealed file\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


CFG = {"paths": {"raw_sealed": "full.csv"}}


def test_holdout_stays_closed_without_the_frozen_tag(repo):
    with pytest.raises(PermissionError, match=FROZEN_TAG):
        open_sealed_holdout(CFG, repo)


def test_holdout_stays_closed_with_uncommitted_changes(repo):
    _git(repo, "tag", FROZEN_TAG)
    (repo / "full.csv").write_text("edited\n")
    with pytest.raises(PermissionError, match="uncommitted"):
        open_sealed_holdout(CFG, repo)


def test_holdout_opens_once_then_stays_closed(repo):
    _git(repo, "tag", FROZEN_TAG)
    # Gate passes; the hash check then rejects this stand-in file, which proves the gate opened.
    with pytest.raises(ValueError, match="SHA-256"):
        open_sealed_holdout(CFG, repo)
    results = repo / RESULTS_FILE
    results.parent.mkdir(parents=True)
    results.write_text("{}")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "test evaluated")
    _git(repo, "tag", "-f", FROZEN_TAG)
    with pytest.raises(PermissionError, match="evaluated once"):
        open_sealed_holdout(CFG, repo)


def _write_plan(root, status):
    plan = {"status": status, "selection_metric": "valid_pr_auc",
            "models": {"xgboost": {"max_depth": [3, 5]}}}
    (root / PLAN_FILE).parent.mkdir(parents=True, exist_ok=True)
    (root / PLAN_FILE).write_text(yaml.safe_dump(plan))
    _git(root, "add", ".")
    _git(root, "commit", "-qm", f"plan {status}")


def test_run_records_require_an_approved_plan(repo):
    _write_plan(repo, "draft")
    with pytest.raises(PermissionError, match="draft"):
        write_run_record("xgboost", {"max_depth": 3}, {"valid_pr_auc": 0.3}, ["f"], 42, repo)


def test_run_records_capture_plan_commit_and_cleanliness(repo):
    _write_plan(repo, "approved")
    first = write_run_record("xgboost", {"max_depth": 3}, {"valid_pr_auc": 0.3}, ["f"], 42, repo)
    rec = json.loads(first.read_text())
    assert rec["dirty"] is False and rec["n_features"] == 1 and len(rec["commit"]) == 40
    assert first.parent == repo / RUNS_DIR
    # An earlier, uncommitted record does not mark the next run as dirty...
    second = write_run_record("xgboost", {"max_depth": 5}, {"valid_pr_auc": 0.3}, ["f"], 42, repo)
    assert json.loads(second.read_text())["dirty"] is False
    # ...but uncommitted code does.
    (repo / "full.csv").write_text("edited\n")
    third = write_run_record("xgboost", {"max_depth": 5}, {"valid_pr_auc": 0.3}, ["f"], 42, repo)
    assert json.loads(third.read_text())["dirty"] is True
    with pytest.raises(ValueError, match="train_/valid_"):
        write_run_record("xgboost", {"max_depth": 3}, {"test_pr_auc": 0.3}, ["f"], 42, repo)
