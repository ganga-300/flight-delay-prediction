"""Member 3 categorical-embedding MLP, following decision D-305."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from flight_delay.features.encoding import FeatureEncoder


class EmbeddingPreprocessor:
    """Fit category vocabularies, imputation and scaling on training rows."""

    def __init__(
        self,
        numeric: list[str],
        categorical: list[str],
        min_count: int = 50,
    ) -> None:
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.encoder = FeatureEncoder(
            self.categorical, self.numeric, min_count=min_count
        )
        self.scaler = StandardScaler()

    def fit(self, train: pd.DataFrame) -> EmbeddingPreprocessor:
        self.encoder.fit(train)
        encoded = self.encoder.transform(train)
        self.scaler.fit(encoded[self.numeric])
        return self

    def transform(
        self, frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        encoded = self.encoder.transform(frame)
        numeric = self.scaler.transform(
            encoded[self.numeric]
        ).astype(np.float32)
        categories = np.column_stack([
            encoded[column].cat.codes.to_numpy(dtype=np.int64)
            for column in self.categorical
        ])

        if not np.isfinite(numeric).all():
            raise ValueError("Numeric inputs contain non-finite values.")
        if (categories < 0).any():
            raise ValueError("Category mapping produced an invalid ID.")

        return numeric, categories

    @property
    def vocabulary_sizes(self) -> list[int]:
        if self.encoder.categories_ is None:
            raise RuntimeError("Fit preprocessing before reading vocabulary sizes.")
        return [
            len(self.encoder.categories_[column])
            for column in self.categorical
        ]

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingPreprocessor:
        return joblib.load(path)


class EmbeddingMLP(nn.Module):
    """Concatenate numeric features and embeddings, then predict one logit."""

    def __init__(
        self,
        numeric_dim: int,
        vocabulary_sizes: list[int],
        embedding_cap: int,
        hidden_dims: tuple[int, int],
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding_dims = [
            max(1, min(embedding_cap, math.ceil(math.sqrt(size))))
            for size in vocabulary_sizes
        ]
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dimension)
            for size, dimension in zip(
                vocabulary_sizes, self.embedding_dims, strict=True
            )
        ])

        input_dim = numeric_dim + sum(self.embedding_dims)
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

    def forward(
        self, numeric: torch.Tensor, categories: torch.Tensor
    ) -> torch.Tensor:
        embedded = [
            layer(categories[:, index])
            for index, layer in enumerate(self.embeddings)
        ]
        combined = torch.cat([numeric, *embedded], dim=1)
        return self.network(combined)
