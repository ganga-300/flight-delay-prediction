"""Member 3 train-only preprocessing, inference and checkpoint selection checks."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from flight_delay.models.embedding_mlp import (  # noqa: E402
    EmbeddingMLP,
    EmbeddingPreprocessor,
)
from flight_delay.models.embedding_training import (  # noqa: E402
    EmbeddingData,
    predict_logits,
    train_embedding,
)
from flight_delay.models.gru_training import evaluate_logits  # noqa: E402


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "distance": [100.0, 200.0, 300.0, 400.0],
            "hour": [6.0, 10.0, 15.0, 20.0],
            "carrier": ["AA", "AA", "UA", "UA"],
            "origin": ["JFK", "EWR", "JFK", "EWR"],
        }
    )


def preprocessor():
    return EmbeddingPreprocessor(["distance", "hour"], ["carrier", "origin"], 2)


def test_unknown_levels_missing_values_and_train_only_state(frame):
    prep = preprocessor().fit(frame)
    means, medians = prep.scaler.mean_.copy(), prep.encoder.medians_.copy()
    categories = {key: list(values) for key, values in prep.encoder.categories_.items()}
    later = frame.iloc[[0]].copy()
    later["distance"], later["carrier"] = np.nan, "UNSEEN"
    numeric, categorical = prep.transform(later)
    assert np.isfinite(numeric).all()
    assert categorical[0, 0] == prep.vocabulary_sizes[0] - 1
    np.testing.assert_array_equal(means, prep.scaler.mean_)
    assert medians == prep.encoder.medians_
    assert categories == prep.encoder.categories_
    assert "UNSEEN" not in categories["carrier"]


def test_rare_levels_are_collapsed_from_training_counts(frame):
    frame.loc[0, "carrier"] = "RARE"
    prep = preprocessor().fit(frame)
    assert "RARE" not in prep.encoder.categories_["carrier"]
    _, categorical = prep.transform(frame)
    assert categorical[0, 0] == prep.vocabulary_sizes[0] - 1


def test_preprocessor_round_trip(frame, tmp_path):
    prep = preprocessor().fit(frame)
    path = tmp_path / "preprocessor.joblib"
    prep.save(path)
    restored = EmbeddingPreprocessor.load(path)
    for before, after in zip(prep.transform(frame), restored.transform(frame), strict=True):
        np.testing.assert_array_equal(before, after)


def test_vocabulary_requires_fit():
    with pytest.raises(RuntimeError, match="Fit preprocessing"):
        _ = preprocessor().vocabulary_sizes


def test_embedding_dimensions_and_checkpoint_round_trip(tmp_path):
    model = EmbeddingMLP(3, [2, 26, 101], 8, (64, 32)).eval()
    assert model.embedding_dims == [2, 6, 8]
    numeric, categorical = torch.randn(5, 3), torch.zeros((5, 3), dtype=torch.long)
    path = tmp_path / "weights.pth"
    torch.save(model.state_dict(), path)
    restored = EmbeddingMLP(3, [2, 26, 101], 8, (64, 32)).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        actual, expected = restored(numeric, categorical), model(numeric, categorical)
    assert actual.shape == (5, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def toy(seed):
    rng = np.random.default_rng(seed)
    numeric = rng.normal(size=(96, 3)).astype(np.float32)
    categories = rng.integers(0, 3, (96, 2), dtype=np.int64)
    labels = (numeric[:, 0] + categories[:, 0] * 0.2 > 0).astype(np.float32)
    return EmbeddingData(numeric, categories, labels)


SETTINGS = {"learning_rate": 0.0003, "hidden_dims": [64, 32], "embedding_cap": 4}


def test_early_stopping_restores_highest_pr_auc_checkpoint():
    train, tune = toy(1), toy(2)
    model, history, best_epoch, metrics = train_embedding(
        train, tune, [3, 3], SETTINGS, seed=42, max_epochs=4, patience=2
    )
    scores = [row["august_pr_auc"] for row in history]
    assert best_epoch == scores.index(max(scores)) + 1
    restored = evaluate_logits(tune.labels, predict_logits(model, tune))
    assert metrics["pr_auc"] == restored["pr_auc"] == max(scores)


def test_refit_has_exact_epochs_and_reproduces_for_same_seed():
    data = toy(3)
    runs = [train_embedding(data, None, [3, 3], SETTINGS, 42, max_epochs=2) for _ in range(2)]
    for _, history, epoch, metrics in runs:
        assert len(history) == epoch == 2
        assert metrics == {}
        assert all("august_pr_auc" not in row for row in history)
    np.testing.assert_array_equal(
        predict_logits(runs[0][0], data), predict_logits(runs[1][0], data)
    )


def test_explicit_d305_defaults_produce_identical_training():
    data = toy(8)
    original = train_embedding(data, None, [3, 3], SETTINGS, 42, max_epochs=1)[0]
    explicit = train_embedding(
        data,
        None,
        [3, 3],
        {**SETTINGS, "dropout": 0.2, "batch_size": 256},
        42,
        max_epochs=1,
    )[0]
    np.testing.assert_array_equal(predict_logits(original, data), predict_logits(explicit, data))


def test_dropout_and_batch_size_control_the_training_run(monkeypatch):
    import flight_delay.models.embedding_training as training

    batches = []
    original_loader = training.DataLoader

    def capture_loader(*args, **kwargs):
        batches.append(kwargs["batch_size"])
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(training, "DataLoader", capture_loader)
    model = train_embedding(
        toy(9),
        None,
        [3, 3],
        {**SETTINGS, "dropout": 0.4, "batch_size": 1024},
        42,
        max_epochs=1,
    )[0]
    assert batches == [1024]
    assert [layer.p for layer in model.network if isinstance(layer, torch.nn.Dropout)] == [0.4, 0.4]
