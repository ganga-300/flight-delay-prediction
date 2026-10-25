"""Synthetic tests for precision-constrained threshold selection."""

import pytest

from flight_delay.models.operating_point import choose_threshold


def test_perfect_separation():
    result = choose_threshold(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
    )

    assert result["feasible"] is True
    assert result["precision"] >= 0.60
    assert result["recall"] == pytest.approx(1.0)


def test_no_feasible_threshold():
    result = choose_threshold(
        [0, 0, 1],
        [0.9, 0.8, 0.1],
        min_precision=0.90,
    )

    assert result["feasible"] is False
    assert result["threshold"] is None


def test_invalid_probabilities():
    with pytest.raises(ValueError):
        choose_threshold([0, 1], [0.2, 1.2])


def test_invalid_precision_target():
    with pytest.raises(ValueError):
        choose_threshold([0, 1], [0.2, 0.8], min_precision=0)


def test_empty_input():
    with pytest.raises(ValueError):
        choose_threshold([], [])


def test_no_positive_targets():
    with pytest.raises(ValueError):
        choose_threshold([0, 0], [0.1, 0.2])
