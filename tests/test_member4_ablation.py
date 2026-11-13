"""Unit tests for Member 4 ablation models and feature slicing utilities."""

import numpy as np
import pytest
import torch

from flight_delay.config import load_config
from flight_delay.features.build import CATEGORICAL, feature_groups, numeric_features
from flight_delay.models.ablation import (
    OneHotMLP,
    get_ablation_numeric_features,
    predict_one_hot_logits,
    train_one_hot_ablation,
)


def test_ablation_numeric_feature_counts():
    cfg = load_config()
    all_num = numeric_features(cfg)
    assert len(all_num) == 49

    no_hist = get_ablation_numeric_features(cfg, "no_history")
    assert len(no_hist) == 20  # 10 calendar + 6 schedule + 4 congestion
    assert not any(c.startswith("hist_") for c in no_hist)

    no_cong = get_ablation_numeric_features(cfg, "no_congestion")
    assert len(no_cong) == 45  # 10 calendar + 6 schedule + 29 historical
    assert not any(c.startswith("sched_dep_origin") for c in no_cong)

    one_hot_num = get_ablation_numeric_features(cfg, "one_hot")
    assert len(one_hot_num) == 49

    no_reg_num = get_ablation_numeric_features(cfg, "no_regularization")
    assert len(no_reg_num) == 49


def test_one_hot_mlp_forward_and_predict():
    batch_size = 16
    input_dim = 1015
    model = OneHotMLP(input_dim=input_dim, hidden_dims=(128, 64), dropout=0.2)

    x = torch.randn(batch_size, input_dim)
    logits = model(x)
    assert logits.shape == (batch_size, 1)

    # Test predict_one_hot_logits
    x_np = x.numpy()
    preds = predict_one_hot_logits(model, x_np, batch_size=8)
    assert preds.shape == (batch_size,)
    assert np.isfinite(preds).all()


def test_train_one_hot_ablation_synthetic():
    rng = np.random.default_rng(42)
    n_train, n_tune = 100, 30
    input_dim = 50

    X_train = rng.standard_normal((n_train, input_dim)).astype(np.float32)
    y_train = rng.integers(0, 2, n_train).astype(np.float32)

    X_tune = rng.standard_normal((n_tune, input_dim)).astype(np.float32)
    y_tune = rng.integers(0, 2, n_tune).astype(np.float32)

    model, history, best_epoch, best_metrics = train_one_hot_ablation(
        X_train,
        y_train,
        X_tune,
        y_tune,
        seed=42,
        learning_rate=0.001,
        hidden_dims=(32, 16),
        dropout=0.1,
        weight_decay=0.0001,
        batch_size=32,
        max_epochs=3,
        patience=2,
    )

    assert len(history) >= 1
    assert 1 <= best_epoch <= 3
    assert "pr_auc" in best_metrics
    assert np.isfinite(best_metrics["pr_auc"])
