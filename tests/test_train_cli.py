"""Tests for the training CLI safety restrictions."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train.py"


@pytest.mark.parametrize("model", ["logreg", "xgboost"])
def test_dry_run_succeeds(model):
    result = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--model",
            model,
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "DRY RUN: PASS" in result.stdout
    assert "Model fitting: not executed" in result.stdout


@pytest.mark.parametrize("model", ["logreg", "xgboost"])
def test_training_without_approval_is_blocked(model):
    result = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--model",
            model,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Actual training is disabled" in result.stderr


def test_invalid_model_is_rejected():
    result = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--model",
            "randomforest",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
