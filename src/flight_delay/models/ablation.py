"""Member 4 ablation study models and feature slicing utilities under D-309."""

from __future__ import annotations

import copy
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flight_delay.features.build import CATEGORICAL, feature_groups, numeric_features
from flight_delay.models.embedding_mlp import EmbeddingMLP, EmbeddingPreprocessor
from flight_delay.models.embedding_training import EmbeddingData, predict_logits
from flight_delay.models.gru_training import evaluate_logits
from flight_delay.models.mlp_training import MLPPreprocessor, set_torch_seed

AblationType = Literal[
    "reference",
    "no_history",
    "no_congestion",
    "one_hot",
    "no_regularization",
]


def get_ablation_numeric_features(cfg: dict, ablation: AblationType) -> list[str]:
    """Return the exact list of numeric feature names for a given ablation."""
    groups = feature_groups(cfg)
    calendar = groups["calendar"]
    schedule_route = groups["schedule_route"]
    congestion = groups["congestion"]
    historical = groups["historical"]

    if ablation == "no_history":
        return calendar + schedule_route + congestion
    elif ablation == "no_congestion":
        return calendar + schedule_route + historical
    else:
        # reference, one_hot, no_regularization all use all 49 numeric features
        return calendar + schedule_route + congestion + historical


class OneHotMLP(nn.Module):
    """Feed-forward MLP baseline consuming one-hot encoded inputs with D-307 architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        first, second = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(input_dim, first),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(first, second),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def predict_one_hot_logits(model: OneHotMLP, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Predict logits for one-hot MLP in fixed-order CPU batches."""
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            stop = start + batch_size
            batch_x = torch.from_numpy(X[start:stop])
            logits = model(batch_x)
            batches.append(logits.squeeze(-1).numpy())
    return np.concatenate(batches)


def train_one_hot_ablation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_tune: np.ndarray | None,
    y_tune: np.ndarray | None,
    seed: int,
    learning_rate: float = 0.0001,
    hidden_dims: tuple[int, int] = (128, 64),
    dropout: float = 0.2,
    weight_decay: float = 0.0001,
    batch_size: int = 1024,
    max_epochs: int = 30,
    patience: int = 5,
) -> tuple[OneHotMLP, list[dict], int, dict]:
    """Train OneHotMLP under the D-301/D-309 protocol."""
    set_torch_seed(seed)
    input_dim = X_train.shape[1]
    model = OneHotMLP(input_dim, hidden_dims=hidden_dims, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train).unsqueeze(-1),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    history, best_metrics = [], {}
    best_epoch, best_pr, stale, best_state = 0, -1.0, 0, None

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            if not torch.isfinite(loss):
                raise ValueError("Training loss became non-finite")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_y)

        record = {"epoch": epoch, "train_loss": total_loss / len(X_train)}
        if X_tune is None or y_tune is None:
            best_epoch = epoch
        else:
            tune_logits = predict_one_hot_logits(model, X_tune)
            metrics = evaluate_logits(y_tune, tune_logits)
            record.update({f"august_{key}": value for key, value in metrics.items()})
            score = metrics["pr_auc"]
            if score > best_pr:
                best_pr, best_epoch, stale = score, epoch, 0
                best_metrics = metrics
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1

        history.append(record)
        if X_tune is not None and stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch, best_metrics
