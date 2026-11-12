"""Member 2 sequence models (D-303): shapes, order sensitivity, training loop, reproducibility."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flight_delay.features.sequences import SEQUENCE_CHANNELS  # noqa: E402
from flight_delay.models.gru import SequenceDelayModel  # noqa: E402
from flight_delay.models.gru_training import (  # noqa: E402
    SequenceData,
    predict_logits,
    train_sequence_model,
)
from flight_delay.models.mlp_training import set_torch_seed  # noqa: E402

C = len(SEQUENCE_CHANNELS)


@pytest.mark.parametrize("encoder", ["gru", "flat"])
def test_output_shape_and_finite_logits(encoder):
    model = SequenceDelayModel(
        static_dim=12, seq_channels=C, seq_len=7, hidden_size=16, encoder=encoder
    )
    out = model(torch.randn(5, 12), torch.randn(5, 7, C))
    assert out.shape == (5, 1)
    assert torch.isfinite(out).all()


def test_unknown_encoder_rejected():
    with pytest.raises(ValueError):
        SequenceDelayModel(4, C, 7, 8, encoder="lstm")


def test_gru_encoding_depends_on_order():
    set_torch_seed(0)
    model = SequenceDelayModel(4, C, 7, 8, encoder="gru").eval()
    seq = torch.randn(3, 7, C)
    with torch.no_grad():
        a = model.encode(seq)
        b = model.encode(torch.flip(seq, dims=[1]))
    assert not torch.allclose(a, b)


def _toy_data(n=600, days=40, length=7, seed=0):
    rng = np.random.default_rng(seed)
    tables = [
        rng.normal(size=(days, 2)).astype(np.float32),  # system
        rng.normal(size=(days, 3, 2)).astype(np.float32),  # origin
        rng.normal(size=(days, 4, 3)).astype(np.float32),  # carrier
        rng.normal(size=(days, 5, 1)).astype(np.float32),  # dest
        rng.normal(size=(days, 2)).astype(np.float32),  # calendar
    ]
    idx = {
        "day": rng.integers(length, days, n),
        "origin": rng.integers(0, 3, n),
        "carrier": rng.integers(0, 4, n),
        "dest": rng.integers(0, 5, n),
    }
    static = rng.normal(size=(n, 6)).astype(np.float32)
    signal = static[:, 0] + tables[0][idx["day"] - 1, 0]
    y = (signal + rng.normal(scale=0.5, size=n) > 0).astype(np.float32)
    return SequenceData(static=static, idx=idx, y=y, tables=tables, length=length)


def test_training_runs_and_restores_best_epoch():
    train, tune = _toy_data(seed=1), _toy_data(seed=2)
    model, history, best_epoch, best = train_sequence_model(
        train, tune, "gru", 8, torch.device("cpu"), max_epochs=4, patience=2, verbose=False
    )
    assert 1 <= best_epoch <= len(history["epoch"]) <= 4
    assert best["pr_auc"] == max(history["valid_pr_auc"])
    logits = predict_logits(model, tune, torch.device("cpu"))
    assert logits.shape == (len(tune),) and np.isfinite(logits).all()


def test_fixed_epoch_refit_has_no_validation():
    train = _toy_data(seed=3)
    _, history, best_epoch, best = train_sequence_model(
        train, None, "flat", 8, torch.device("cpu"), max_epochs=3, verbose=False
    )
    assert best_epoch == 3 and history["epoch"] == [1, 2, 3]
    assert history["valid_pr_auc"] == [] and best == {}


def test_same_seed_same_model():
    train = _toy_data(seed=4)
    runs = [
        train_sequence_model(
            train, None, "gru", 8, torch.device("cpu"), max_epochs=2, seed=7, verbose=False
        )[0]
        for _ in range(2)
    ]
    a, b = (predict_logits(m, train, torch.device("cpu")) for m in runs)
    np.testing.assert_array_equal(a, b)
