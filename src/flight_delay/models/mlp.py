"""Multi-Layer Perceptron (MLP) baseline model for flight delay prediction.

Phase 2 - Member 1: Simple MLP baseline using PyTorch and One-Hot Encoding.
Architecture:
    Input -> Linear(128) -> ReLU -> Dropout(0.3) -> Linear(64) -> ReLU -> Dropout(0.2) -> Linear(1)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """Feed-forward Multi-Layer Perceptron (MLP) binary classifier.

    Architecture:
        * Input Layer: preprocessed numerical and one-hot categorical features
        * Hidden Layer 1: 128 neurons, ReLU activation, Dropout = 0.3
        * Hidden Layer 2: 64 neurons, ReLU activation, Dropout = 0.2
        * Output Layer: 1 neuron, returning raw logits (no sigmoid in forward)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        dropout1: float = 0.3,
        dropout2: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout1 = dropout1
        self.dropout2 = dropout2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing raw logits.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Logits tensor of shape (batch_size, 1).
        """
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute predicted probabilities using sigmoid activation.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Probability tensor of shape (batch_size, 1) in [0, 1].
        """
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Compute binary class predictions based on decision threshold.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            threshold: Probability threshold for positive class classification.

        Returns:
            Binary predictions tensor of shape (batch_size, 1) with values in {0, 1}.
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).long()
