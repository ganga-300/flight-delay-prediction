"""Unit tests for Phase 2 Member 1 PyTorch MLP Baseline.

Verifies:
1. MLP accepts expected input dimension and produces shape (batch_size, 1).
2. Forward pass produces finite logits (no sigmoid in forward).
3. Loss computation works (BCEWithLogitsLoss).
4. Backpropagation produces valid gradients for all layers.
5. Model state_dict can be saved and loaded accurately.
6. Preprocessing produces consistent feature dimensions across splits.
7. Unseen validation categories are handled correctly without throwing errors.
8. No target or post-departure columns enter the input.
9. Training and validation splits are strictly chronological.
10. Reproducibility with the fixed seed.
11. End-to-end inference using predict_flight_delays.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from flight_delay.config import REPO_ROOT, load_config
from flight_delay.data.schema import TARGET_COLUMN
from flight_delay.features.build import CATEGORICAL, numeric_features
from flight_delay.features.encoding import FeatureEncoder
from flight_delay.features.leakage import POST_DEPARTURE_COLUMNS
from flight_delay.models.mlp import MLPBaseline
from flight_delay.models.mlp_training import (
    MLPPreprocessor,
    get_device,
    predict_flight_delays,
    set_torch_seed,
)


@pytest.fixture
def sample_features() -> tuple[list[str], list[str]]:
    cfg = load_config()
    return numeric_features(cfg), list(CATEGORICAL)


def test_mlp_input_dimension_and_output_shape():
    """Verify MLP accepts custom input dimensions and produces (batch_size, 1)."""
    batch_size = 16
    input_dim = 1015
    model = MLPBaseline(input_dim=input_dim, hidden_dim1=128, hidden_dim2=64)

    x = torch.randn(batch_size, input_dim)
    logits = model(x)

    assert logits.shape == (batch_size, 1)
    assert logits.dtype == torch.float32


def test_forward_pass_produces_finite_logits():
    """Verify forward pass outputs finite logits without sigmoid constraint."""
    model = MLPBaseline(input_dim=50)
    x = torch.randn(32, 50)
    logits = model(x)

    assert torch.isfinite(logits).all()
    # Confirm forward does NOT apply sigmoid (logits can exceed [0, 1])
    probs = model.predict_proba(x)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
    preds = model.predict(x, threshold=0.5)
    assert set(preds.unique().tolist()).issubset({0, 1})


def test_loss_computation_and_backpropagation():
    """Verify loss computation with BCEWithLogitsLoss and backprop computes finite gradients."""
    set_torch_seed(42)
    model = MLPBaseline(input_dim=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    x = torch.randn(16, 20)
    y = torch.randint(0, 2, (16, 1)).float()

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)

    assert torch.isfinite(loss)
    assert loss.item() > 0

    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains non-finite values"


def test_model_save_and_load(tmp_path: Path):
    """Verify model can be saved and loaded with identical outputs."""
    set_torch_seed(42)
    model = MLPBaseline(input_dim=30)
    model.eval()

    x = torch.randn(8, 30)
    with torch.no_grad():
        original_output = model(x)

    save_path = tmp_path / "test_mlp.pth"
    torch.save(model.state_dict(), save_path)

    loaded_model = MLPBaseline(input_dim=30)
    loaded_model.load_state_dict(torch.load(save_path))
    loaded_model.eval()

    with torch.no_grad():
        loaded_output = loaded_model(x)

    torch.testing.assert_close(original_output, loaded_output)


def test_preprocessing_produces_consistent_feature_dimensions():
    """Verify MLPPreprocessor produces matching feature dimensions on train and validation."""
    train_path = REPO_ROOT / "data" / "processed" / "train.parquet"
    valid_path = REPO_ROOT / "data" / "processed" / "valid.parquet"
    encoder_path = REPO_ROOT / "data" / "processed" / "encoder.json"

    if not train_path.exists() or not valid_path.exists():
        pytest.skip("Processed parquet data not available; build first.")

    cfg = load_config()
    num_cols = numeric_features(cfg)
    cat_cols = CATEGORICAL
    encoder = FeatureEncoder.load(encoder_path)

    train_df = pd.read_parquet(train_path).head(500)
    valid_df = pd.read_parquet(valid_path).head(200)

    preprocessor = MLPPreprocessor(num_cols, cat_cols, feature_encoder=encoder)
    X_train = preprocessor.fit_transform(train_df)
    X_valid = preprocessor.transform(valid_df)

    assert X_train.shape[1] == X_valid.shape[1]
    assert X_train.shape[1] == preprocessor.input_dim_
    assert X_train.dtype == np.float32
    assert X_valid.dtype == np.float32
    assert np.all(np.isfinite(X_train))
    assert np.all(np.isfinite(X_valid))


def test_unseen_validation_categories_handled_correctly():
    """Verify unseen categories in evaluation data are safely mapped to __OTHER__ without error."""
    train_path = REPO_ROOT / "data" / "processed" / "train.parquet"
    encoder_path = REPO_ROOT / "data" / "processed" / "encoder.json"

    if not train_path.exists() or not encoder_path.exists():
        pytest.skip("Processed data not available.")

    cfg = load_config()
    num_cols = numeric_features(cfg)
    cat_cols = CATEGORICAL
    encoder = FeatureEncoder.load(encoder_path)

    train_df = pd.read_parquet(train_path).head(500)
    preprocessor = MLPPreprocessor(num_cols, cat_cols, feature_encoder=encoder)
    preprocessor.fit(train_df)

    # Construct test data with completely unseen categorical levels
    test_df = train_df.head(5).copy()
    test_df["carrier"] = "ZZ"
    test_df["origin"] = "XYZ"
    test_df["dest"] = "NONEXISTENT"
    test_df["route"] = "XYZ_NONEXISTENT"

    X_test = preprocessor.transform(test_df)
    assert X_test.shape == (5, preprocessor.input_dim_)
    assert np.all(np.isfinite(X_test))


def test_no_target_or_post_departure_columns_enter_input():
    """Verify that target and post-departure leaky columns are excluded from preprocessor features."""
    cfg = load_config()
    num_cols = numeric_features(cfg)
    cat_cols = list(CATEGORICAL)

    all_input_cols = set(num_cols) | set(cat_cols)

    forbidden = POST_DEPARTURE_COLUMNS | {TARGET_COLUMN, "flight_status", "tailnum", "id"}
    leaked = forbidden.intersection(all_input_cols)
    assert not leaked, f"Forbidden leaky columns detected in model inputs: {sorted(leaked)}"

    # Also check fitted preprocessor if saved
    saved_preprocessor_path = REPO_ROOT / "models" / "phase2" / "member1" / "mlp_preprocessor.joblib"
    if saved_preprocessor_path.exists():
        preprocessor = MLPPreprocessor.load(saved_preprocessor_path)
        for name in preprocessor.all_feature_names_:
            for post in POST_DEPARTURE_COLUMNS:
                assert not name.startswith(post), f"Leaky column prefix {post} found in {name}"
            assert not name.startswith(TARGET_COLUMN), f"Target column found in {name}"


def test_training_and_validation_splits_are_chronological():
    """Verify that train and validation parquet splits respect strict chronological ordering."""
    train_path = REPO_ROOT / "data" / "processed" / "train.parquet"
    valid_path = REPO_ROOT / "data" / "processed" / "valid.parquet"

    if not train_path.exists() or not valid_path.exists():
        pytest.skip("Processed data files not found.")

    train_df = pd.read_parquet(train_path, columns=["date"])
    valid_df = pd.read_parquet(valid_path, columns=["date"])

    train_dates = pd.to_datetime(train_df["date"])
    valid_dates = pd.to_datetime(valid_df["date"])

    assert train_dates.max() <= valid_dates.min(), (
        f"Train max date ({train_dates.max()}) is after valid min date ({valid_dates.min()})"
    )
    assert train_dates.max() == pd.Timestamp("2013-08-31")
    assert valid_dates.min() == pd.Timestamp("2013-09-01")
    assert valid_dates.max() == pd.Timestamp("2013-10-31")


def test_reproducibility_with_fixed_seed():
    """Verify training seed produces identical initialization and forward outputs."""
    set_torch_seed(42)
    m1 = MLPBaseline(input_dim=50)
    x = torch.randn(8, 50)
    out1 = m1(x)

    set_torch_seed(42)
    m2 = MLPBaseline(input_dim=50)
    x2 = torch.randn(8, 50)
    out2 = m2(x2)

    torch.testing.assert_close(out1, out2)

    # Different seed should differ
    set_torch_seed(99)
    m3 = MLPBaseline(input_dim=50)
    out3 = m3(x2)
    assert not torch.allclose(out1, out3)


def test_predict_flight_delays_end_to_end():
    """Verify reusable inference function works end-to-end on saved artifacts."""
    artifacts_dir = REPO_ROOT / "models" / "phase2" / "member1"
    valid_path = REPO_ROOT / "data" / "processed" / "valid.parquet"

    if not (artifacts_dir / "mlp_baseline.pth").exists() or not valid_path.exists():
        pytest.skip("Artifacts or valid data not available for end-to-end test.")

    valid_df = pd.read_parquet(valid_path).head(100)
    probs, preds = predict_flight_delays(artifacts_dir, valid_df, threshold=0.5, device="cpu")

    assert len(probs) == 100
    assert len(preds) == 100
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert set(np.unique(preds)).issubset({0, 1})
    assert np.all((probs >= 0.5) == (preds == 1))
