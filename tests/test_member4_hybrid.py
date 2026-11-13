"""Unit tests for Member 4 hybrid DL + XGBoost pipeline."""

import numpy as np
import pandas as pd
import pytest
import torch

from flight_delay.models.embedding_mlp import EmbeddingMLP
from flight_delay.models.embedding_training import EmbeddingData
from flight_delay.models.hybrid import (
    build_hybrid_feature_matrix,
    extract_neural_representations,
    get_default_xgboost_params,
    train_hybrid_xgboost,
    train_plain_xgboost,
)


def test_extract_neural_representations_and_hybrid_matrix():
    n_samples = 32
    numeric_dim = 49
    vocab_sizes = [16, 4, 20]  # 3 categorical columns
    embedding_cap = 4
    hidden_dims = (128, 64)

    model = EmbeddingMLP(
        numeric_dim=numeric_dim,
        vocabulary_sizes=vocab_sizes,
        embedding_cap=embedding_cap,
        hidden_dims=hidden_dims,
        dropout=0.2,
    )

    rng = np.random.default_rng(42)
    numeric = rng.standard_normal((n_samples, numeric_dim)).astype(np.float32)
    categories = np.column_stack(
        [rng.integers(0, size, n_samples) for size in vocab_sizes]
    ).astype(np.int64)
    labels = rng.integers(0, 2, n_samples).astype(np.float32)

    data = EmbeddingData(numeric, categories, labels)

    representations = extract_neural_representations(model, data, batch_size=16)
    assert representations.shape == (n_samples, 64)
    assert np.isfinite(representations).all()

    # Build hybrid matrix
    Z = build_hybrid_feature_matrix(numeric, representations)
    assert Z.shape == (n_samples, 49 + 64)
    assert Z.dtype == np.float32
    assert np.isfinite(Z).all()


def test_train_hybrid_xgboost_synthetic():
    rng = np.random.default_rng(42)
    n_train, n_tune = 80, 20
    feature_dim = 113

    X_train = rng.standard_normal((n_train, feature_dim)).astype(np.float32)
    y_train = rng.integers(0, 2, n_train).astype(np.float32)

    X_tune = rng.standard_normal((n_tune, feature_dim)).astype(np.float32)
    y_tune = rng.integers(0, 2, n_tune).astype(np.float32)

    params = get_default_xgboost_params(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=10,
        early_stopping_rounds=5,
        seed=42,
    )

    model, effective_trees = train_hybrid_xgboost(X_train, y_train, X_tune, y_tune, params)
    assert 1 <= effective_trees <= 10
    preds = model.predict_proba(X_tune)[:, 1]
    assert preds.shape == (n_tune,)
    assert np.all((preds >= 0.0) & (preds <= 1.0))


def test_train_plain_xgboost_synthetic():
    rng = np.random.default_rng(42)
    n_samples = 60

    df = pd.DataFrame(
        {
            "num1": rng.standard_normal(n_samples),
            "num2": rng.standard_normal(n_samples),
            "cat1": pd.Categorical(rng.choice(["A", "B", "C"], n_samples)),
        }
    )
    y = rng.integers(0, 2, n_samples)

    params = get_default_xgboost_params(
        max_depth=2,
        learning_rate=0.1,
        n_estimators=5,
        early_stopping_rounds=None,
        seed=42,
        enable_categorical=True,
    )

    model, trees = train_plain_xgboost(df, y, None, None, params)
    assert trees == 5
    probs = model.predict_proba(df)[:, 1]
    assert len(probs) == n_samples
