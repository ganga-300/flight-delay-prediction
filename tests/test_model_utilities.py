"""Tests for model calibration and persistence."""

from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from flight_delay.models.calibration import calibrate
from flight_delay.models.persistence import load_model, save_model


@pytest.fixture
def fitted_model_and_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)

    model = LogisticRegression(random_state=42)
    model.fit(X[:120], y[:120])

    return model, X[120:], y[120:]


def test_sigmoid_calibration(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    original_coef = model.coef_.copy()

    calibrated = calibrate(model, X, y, method="sigmoid")
    probabilities = calibrated.predict_proba(X)[:, 1]

    assert probabilities.shape == (len(X),)
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_array_equal(model.coef_, original_coef)


def test_isotonic_calibration(fitted_model_and_data):
    model, X, y = fitted_model_and_data

    calibrated = calibrate(model, X, y, method="isotonic")
    probabilities = calibrated.predict_proba(X)[:, 1]

    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_invalid_calibration_method(fitted_model_and_data):
    model, X, y = fitted_model_and_data

    with pytest.raises(ValueError):
        calibrate(model, X, y, method="invalid")


def test_model_save_load(tmp_path: Path, fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    path = tmp_path / "nested" / "model.joblib"

    save_model(model, path)
    restored = load_model(path)

    assert path.exists()
    np.testing.assert_allclose(
        model.predict_proba(X),
        restored.predict_proba(X),
    )


def test_missing_model_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")


def test_saved_artifact_is_joblib(tmp_path: Path, fitted_model_and_data):
    model, _, _ = fitted_model_and_data
    path = tmp_path / "model.joblib"

    save_model(model, path)

    assert isinstance(joblib.load(path), LogisticRegression)
