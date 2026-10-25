"""Tests for experiment artifact persistence."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from flight_delay.features.encoding import FeatureEncoder
from flight_delay.models.artifacts import save_experiment_artifacts
from flight_delay.models.persistence import load_model


@pytest.fixture
def experiment():
    rng = np.random.default_rng(42)

    X = pd.DataFrame({"feature": rng.normal(size=40)})
    y = np.array([0, 1] * 20)

    model = LogisticRegression().fit(X, y)

    encoder = FeatureEncoder(
        categorical=[f"cat_{i}" for i in range(10)],
        numeric=[f"num_{i}" for i in range(49)],
    )

    result = {
        "calibrated_model": model,
        "selection": {
            "model_name": "logreg",
            "parameters": {"C": 1.0},
            "validation_pr_auc": 0.70,
            "comparison": {"logreg": 0.70, "xgboost": 0.65},
        },
        "logistic": {
            "results": [{"model": "logreg", "pr_auc": 0.70}] * 8,
        },
        "xgboost": {
            "results": [{"model": "xgboost", "pr_auc": 0.65}] * 4,
        },
        "operating_point": {
            "feasible": True,
            "threshold": 0.6,
            "precision": 0.65,
            "recall": 0.40,
        },
    }

    return model, encoder, result, X


def test_artifact_files_created(tmp_path, experiment):
    _, encoder, result, _ = experiment

    paths = save_experiment_artifacts(result, encoder, tmp_path)

    assert paths["model"].exists()
    assert paths["encoder"].exists()
    assert paths["results"].exists()


def test_saved_model_loads(tmp_path, experiment):
    model, encoder, result, X = experiment

    paths = save_experiment_artifacts(result, encoder, tmp_path)
    restored = load_model(paths["model"])

    np.testing.assert_allclose(
        model.predict_proba(X),
        restored.predict_proba(X),
    )


def test_results_metadata(tmp_path, experiment):
    _, encoder, result, _ = experiment

    paths = save_experiment_artifacts(result, encoder, tmp_path)
    report = json.loads(paths["results"].read_text())

    assert report["protocol"] == "D-201"
    assert report["test_evaluated"] is False
    assert report["feature_count"] == 59
    assert report["selected_model"] == "logreg"
    assert len(report["logistic_experiments"]) == 8
    assert len(report["xgboost_experiments"]) == 4


def test_rejects_wrong_feature_count(tmp_path, experiment):
    _, encoder, result, _ = experiment
    encoder.numeric.pop()

    with pytest.raises(ValueError):
        save_experiment_artifacts(result, encoder, tmp_path)
