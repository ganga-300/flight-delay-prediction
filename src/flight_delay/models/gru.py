"""Member 2 sequence models (docs/decision_log.md, D-303).

Both models share the static branch and the head; only the sequence encoder differs:

* ``encoder="gru"``: a 1-layer GRU reads the L daily steps in order; its final state is used.
* ``encoder="flat"`` (control, no recurrence): the same L x C values are flattened and passed
  through Linear(hidden) -> ReLU, so the model sees the same inputs but not their order as a
  sequence.

Head: [static features ; sequence encoding] -> Linear(64) -> ReLU -> Dropout(0.2) -> Linear(1),
returning raw logits (the sigmoid is applied only when producing probabilities).
"""

from __future__ import annotations

import torch
import torch.nn as nn

ENCODERS = ("gru", "flat")


class SequenceDelayModel(nn.Module):
    def __init__(
        self,
        static_dim: int,
        seq_channels: int,
        seq_len: int,
        hidden_size: int,
        encoder: str = "gru",
        head_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError(f"encoder must be one of {ENCODERS}, got {encoder!r}")
        self.encoder_type = encoder
        self.static_dim = static_dim
        self.seq_channels = seq_channels
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        if encoder == "gru":
            self.encoder = nn.GRU(seq_channels, hidden_size, num_layers=1, batch_first=True)
        else:
            self.encoder = nn.Sequential(
                nn.Flatten(), nn.Linear(seq_len * seq_channels, hidden_size), nn.ReLU()
            )
        self.head = nn.Sequential(
            nn.Linear(static_dim + hidden_size, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def encode(self, x_seq: torch.Tensor) -> torch.Tensor:
        """[B, L, C] sequences -> [B, hidden_size]."""
        if self.encoder_type == "gru":
            _, h_n = self.encoder(x_seq)
            return h_n[-1]
        return self.encoder(x_seq)

    def forward(self, x_static: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
        """Logits of shape [B, 1]."""
        return self.head(torch.cat([x_static, self.encode(x_seq)], dim=1))
