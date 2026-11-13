"""Deterministic CPU training for Member 3 under D-305 and the D-307 amendment."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flight_delay.models.embedding_mlp import EmbeddingMLP
from flight_delay.models.gru_training import evaluate_logits
from flight_delay.models.mlp_training import set_torch_seed


@dataclass
class EmbeddingData:
    numeric: np.ndarray
    categories: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


def predict_logits(model: EmbeddingMLP, data: EmbeddingData) -> np.ndarray:
    """Predict in fixed-order CPU batches without updating model parameters."""
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(data), 4096):
            stop = start + 4096
            logits = model(
                torch.from_numpy(data.numeric[start:stop]),
                torch.from_numpy(data.categories[start:stop]),
            )
            batches.append(logits.squeeze(-1).numpy())
    return np.concatenate(batches)


def train_embedding(
    train: EmbeddingData,
    tune: EmbeddingData | None,
    vocabulary_sizes: list[int],
    settings: dict,
    seed: int,
    max_epochs: int = 30,
    patience: int = 5,
) -> tuple[EmbeddingMLP, list[dict], int, dict]:
    """Select epochs on August PR-AUC, or perform a fixed-epoch final refit."""
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    set_torch_seed(seed)
    model = EmbeddingMLP(
        train.numeric.shape[1],
        vocabulary_sizes,
        settings["embedding_cap"],
        tuple(settings["hidden_dims"]),
        dropout=settings.get("dropout", 0.2),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings["learning_rate"], weight_decay=0.0001
    )
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train.numeric),
            torch.from_numpy(train.categories),
            torch.from_numpy(train.labels).unsqueeze(-1),
        ),
        batch_size=settings.get("batch_size", 256),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    history, best_metrics = [], {}
    best_epoch, best_pr, stale, best_state = 0, -1.0, 0, None
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for numeric, categories, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(numeric, categories), labels)
            if not torch.isfinite(loss):
                raise ValueError("Training loss became non-finite")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
        record = {"epoch": epoch, "train_loss": total_loss / len(train)}
        if tune is None:
            best_epoch = epoch
            print(f"  refit epoch {epoch:02d}: loss={record['train_loss']:.5f}", flush=True)
        else:
            metrics = evaluate_logits(tune.labels, predict_logits(model, tune))
            record.update({f"august_{key}": value for key, value in metrics.items()})
            score = metrics["pr_auc"]
            if score > best_pr:
                best_pr, best_epoch, stale = score, epoch, 0
                best_metrics = metrics
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            print(
                f"  epoch {epoch:02d}: loss={record['train_loss']:.5f}, "
                f"August PR-AUC={score:.5f}, best_epoch={best_epoch}",
                flush=True,
            )
        history.append(record)
        if tune is not None and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch, best_metrics
