"""Temporal encoders reused by the integrated Jiadong STGNN module."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """LSTM encoder that returns the last hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.dropout(out[:, -1, :])


class GRUEncoder(nn.Module):
    """GRU encoder that returns the last hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.dropout(out[:, -1, :])


class MHAEncoder(nn.Module):
    """Multi-head self-attention encoder with positional encoding."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        pooling: str = "attention",
        max_len: int = 64,
    ):
        super().__init__()
        self.output_dim = hidden_dim
        self.pooling = pooling

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if hidden_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        if pooling == "attention":
            self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, _ = x.shape
        h = self.input_proj(x) + self.pe[:, :steps, :]
        attn_out, _ = self.attn(h, h, h)
        h = self.norm(h + attn_out)

        if self.pooling == "mean":
            out = h.mean(dim=1)
        else:
            query = self.pool_query.expand(batch_size, -1, -1)
            pooled, _ = self.attn(query, h, h)
            out = pooled.squeeze(1)

        return self.dropout(out)


TEMPORAL_REGISTRY = {
    "lstm": LSTMEncoder,
    "gru": GRUEncoder,
    "mha": MHAEncoder,
}


def build_temporal_encoder(
    name: str,
    input_dim: int,
    hidden_dim: int,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    """Build a temporal encoder by name."""
    name = name.lower()
    if name not in TEMPORAL_REGISTRY:
        raise ValueError(
            f"Unknown temporal encoder '{name}'. "
            f"Choose from {list(TEMPORAL_REGISTRY.keys())}"
        )
    encoder_cls = TEMPORAL_REGISTRY[name]
    return encoder_cls(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout, **kwargs)

