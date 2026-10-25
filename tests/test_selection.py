"""Tests for validation-only winner selection."""

import pytest

from flight_delay.models.selection import select_best_model


def candidate(score):
    return {
        "best_model": object(),
        "best_params": {"example": 1},
        "best_pr_auc": score,
    }


def test_xgboost_wins():
    result = select_best_model(candidate(0.60), candidate(0.70))
    assert result["model_name"] == "xgboost"


def test_logistic_wins():
    result = select_best_model(candidate(0.80), candidate(0.70))
    assert result["model_name"] == "logreg"


def test_tie_prefers_logistic():
    result = select_best_model(candidate(0.70), candidate(0.70))
    assert result["model_name"] == "logreg"


def test_rejects_invalid_score():
    with pytest.raises(ValueError):
        select_best_model(candidate(float("nan")), candidate(0.70))


def test_rejects_missing_model():
    invalid = candidate(0.90)
    invalid["best_model"] = None

    with pytest.raises(ValueError):
        select_best_model(invalid, candidate(0.70))
