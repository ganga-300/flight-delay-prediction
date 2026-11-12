"""Training and evaluation for the Member 2 sequence models (D-303).

Same loop as Member 1 (``mlp_training.py``): Adam, unweighted BCEWithLogitsLoss, early stopping
on the tuning period's PR-AUC, best checkpoint restored. With ``tune=None`` the model trains
for exactly ``max_epochs`` epochs (the January-August refit).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from flight_delay.evaluation.metrics import brier, expected_calibration_error, pr_auc, roc_auc
from flight_delay.features.sequences import SEQUENCE_CHANNELS, gather_sequences
from flight_delay.models.gru import SequenceDelayModel
from flight_delay.models.mlp_training import set_torch_seed


@dataclass
class SequenceData:
    """Flights of one period: static features, sequence indices into scaled tables, labels."""

    static: np.ndarray  # [N, D] float32
    idx: dict[str, np.ndarray]  # day / origin / carrier / dest indices, each [N]
    y: np.ndarray  # [N] float32
    tables: list[np.ndarray]  # scaled daily arrays (SequenceScaler.transform_tables)
    length: int

    def __len__(self) -> int:
        return len(self.y)

    def batch(self, rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        part = {key: val[rows] for key, val in self.idx.items()}
        x_seq = gather_sequences(self.tables, part, self.length)
        return torch.from_numpy(self.static[rows]), torch.from_numpy(x_seq)


def predict_logits(
    model: SequenceDelayModel, data: SequenceData, device: torch.device, batch_size: int = 4096
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            rows = np.arange(start, min(start + batch_size, len(data)))
            xs, xq = data.batch(rows)
            out.append(model(xs.to(device), xq.to(device)).squeeze(-1).cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def evaluate_logits(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    p = sigmoid(logits)
    loss = nn.BCEWithLogitsLoss()(
        torch.from_numpy(logits.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
    )
    return {
        "pr_auc": float(pr_auc(y, p)),
        "roc_auc": float(roc_auc(y, p)),
        "brier_score": float(brier(y, p)),
        "ece_10": float(expected_calibration_error(y, p, n_bins=10)),
        "loss": float(loss.item()),
        "mean_predicted_probability": float(np.mean(p)),
        "positive_rate": float(np.mean(y)),
    }


def train_sequence_model(
    train: SequenceData,
    tune: SequenceData | None,
    encoder: str,
    hidden_size: int,
    device: torch.device,
    learning_rate: float = 0.001,
    batch_size: int = 256,
    max_epochs: int = 30,
    patience: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[SequenceDelayModel, dict[str, list[Any]], int, dict[str, float]]:
    """Returns (model, history, best_epoch, tune metrics at the best epoch)."""
    set_torch_seed(seed)
    model = SequenceDelayModel(
        static_dim=train.static.shape[1],
        seq_channels=len(SEQUENCE_CHANNELS),
        seq_len=train.length,
        hidden_size=hidden_size,
        encoder=encoder,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator()
    generator.manual_seed(seed)

    history: dict[str, list[Any]] = {
        "epoch": [],
        "train_loss": [],
        "valid_loss": [],
        "valid_pr_auc": [],
        "valid_roc_auc": [],
        "valid_brier": [],
        "valid_ece": [],
    }
    best_pr, best_epoch, best_state, best_metrics, stale = -1.0, 0, None, {}, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator).numpy()
        total, batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            xs, xq = train.batch(rows)
            yb = torch.from_numpy(train.y[rows]).unsqueeze(-1).to(device)
            optimizer.zero_grad()
            loss = criterion(model(xs.to(device), xq.to(device)), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
        train_loss = total / max(1, batches)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)

        if tune is None:
            best_epoch = epoch
            if verbose:
                print(f"  epoch {epoch:2d}/{max_epochs} | train loss {train_loss:.5f}", flush=True)
            continue

        m = evaluate_logits(tune.y, predict_logits(model, tune, device))
        history["valid_loss"].append(m["loss"])
        history["valid_pr_auc"].append(m["pr_auc"])
        history["valid_roc_auc"].append(m["roc_auc"])
        history["valid_brier"].append(m["brier_score"])
        history["valid_ece"].append(m["ece_10"])
        improved = m["pr_auc"] > best_pr
        if verbose:
            print(
                f"  epoch {epoch:2d}/{max_epochs} | train loss {train_loss:.5f} | "
                f"tune loss {m['loss']:.5f} | tune PR-AUC {m['pr_auc']:.5f}"
                + (" *" if improved else ""),
                flush=True,
            )
        if improved:
            best_pr, best_epoch, best_metrics, stale = m["pr_auc"], epoch, m, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch, best_metrics
