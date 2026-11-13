"""Member 4 Hybrid Deep Learning + XGBoost pipeline under decision D-309."""

from __future__ import annotations

import os

# Limit OpenMP threads to 1 for Apple Silicon compatibility
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from flight_delay.models.embedding_mlp import EmbeddingMLP
from flight_delay.models.embedding_training import EmbeddingData


def extract_neural_representations(
    model: EmbeddingMLP,
    data: EmbeddingData,
    batch_size: int = 4096,
) -> np.ndarray:
    """Extract 64-dimensional latent activations from penultimate layer (after ReLU).

    The D-307 architecture network is:
      [0] Linear(input_dim, 128)
      [1] ReLU()
      [2] Dropout(p)
      [3] Linear(128, 64)
      [4] ReLU()
      [5] Dropout(p)
      [6] Linear(64, 1)

    Activations after [4] (second hidden layer post-ReLU) represent the 64-dimensional
    latent features combining numerical and categorical embedding representations.
    """
    model.eval()
    representations = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            stop = start + batch_size
            num = torch.from_numpy(data.numeric[start:stop])
            cat = torch.from_numpy(data.categories[start:stop])

            embedded = [layer(cat[:, idx]) for idx, layer in enumerate(model.embeddings)]
            combined = torch.cat([num, *embedded], dim=1)

            # Pass through first hidden block (linear + relu)
            h1 = model.network[1](model.network[0](combined))
            # Pass through second hidden block (linear + relu)
            h2 = model.network[4](model.network[3](h1))

            representations.append(h2.cpu().numpy())

    return np.concatenate(representations, axis=0).astype(np.float32)


def build_hybrid_feature_matrix(
    numeric_features: np.ndarray,
    neural_representations: np.ndarray,
) -> np.ndarray:
    """Concatenate original scaled numeric features with 64-d neural representations.

    Returns a dense float32 array of shape (N, 49 + 64) = (N, 113).
    """
    if len(numeric_features) != len(neural_representations):
        raise ValueError("Row count mismatch between numeric features and neural representations.")
    return np.hstack([numeric_features, neural_representations]).astype(np.float32)


def get_default_xgboost_params(
    max_depth: int = 5,
    learning_rate: float = 0.10,
    n_estimators: int = 1000,
    early_stopping_rounds: int | None = 50,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    scale_pos_weight: float = 1.0,
    seed: int = 42,
    enable_categorical: bool = False,
) -> dict:
    """Return XGBoost parameters consistent with D-201 and D-309."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "scale_pos_weight": scale_pos_weight,
        "random_state": seed,
        "n_jobs": 1,
        "enable_categorical": enable_categorical,
    }
    if early_stopping_rounds is not None:
        params["early_stopping_rounds"] = early_stopping_rounds
    return params


def train_hybrid_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_tune: np.ndarray | None,
    y_tune: np.ndarray | None,
    params: dict,
) -> tuple[XGBClassifier, int]:
    """Train XGBoost on continuous hybrid feature matrix.

    Returns fitted model and effective number of trees.
    """
    model = XGBClassifier(**params)
    if X_tune is not None and y_tune is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_tune, y_tune)],
            verbose=False,
        )
        effective_trees = int(model.best_iteration) + 1
    else:
        model.fit(X_train, y_train, verbose=False)
        effective_trees = int(params.get("n_estimators", 100))
    return model, effective_trees


def train_plain_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_tune: pd.DataFrame | None,
    y_tune: np.ndarray | None,
    params: dict,
) -> tuple[XGBClassifier, int]:
    """Train plain XGBoost directly on the 59 original features with native categorical support."""
    plain_params = params.copy()
    plain_params["enable_categorical"] = True
    model = XGBClassifier(**plain_params)
    if X_tune is not None and y_tune is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_tune, y_tune)],
            verbose=False,
        )
        effective_trees = int(model.best_iteration) + 1
    else:
        model.fit(X_train, y_train, verbose=False)
        effective_trees = int(params.get("n_estimators", 100))
    return model, effective_trees
