"""Spatiotemporal GNN architecture integrated from Jiadong Lu's work."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.experimental.jiadong_stgnn.config import (
    DROPOUT,
    GNN_HIDDEN_DIM,
    MHA_NUM_HEADS,
    NUM_FEATURES,
    NUM_GNN_LAYERS,
    NUM_REGIONS,
    TEMPORAL_HIDDEN_DIM,
)
from src.experimental.jiadong_stgnn.models.temporal_modules import build_temporal_encoder


class GCNLayer(nn.Module):
    """Single GCN layer: H' = A_norm @ H @ W + b."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(adjacency, hidden)
        return self.linear(support)


class GNNBlock(nn.Module):
    """Multi-layer GCN with ReLU + dropout between layers."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        for idx in range(num_layers):
            input_dim = in_dim if idx == 0 else hidden_dim
            layers.append(GCNLayer(input_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = features
        for idx, layer in enumerate(self.layers):
            hidden = layer(hidden, adjacency)
            if idx < len(self.layers) - 1:
                hidden = F.relu(hidden)
                hidden = self.dropout(hidden)
        return hidden


class STGNN(nn.Module):
    """Spatiotemporal GNN with configurable temporal encoder."""

    def __init__(
        self,
        temporal_type: str = "lstm",
        num_features: int = NUM_FEATURES,
        num_regions: int = NUM_REGIONS,
        gnn_hidden: int = GNN_HIDDEN_DIM,
        temporal_hidden: int = TEMPORAL_HIDDEN_DIM,
        num_gnn_layers: int = NUM_GNN_LAYERS,
        dropout: float = DROPOUT,
        mha_num_heads: int = MHA_NUM_HEADS,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.gnn_hidden = gnn_hidden

        self.gnn = GNNBlock(num_features, gnn_hidden, num_gnn_layers, dropout)

        extra_kwargs = {}
        if temporal_type == "mha":
            extra_kwargs["num_heads"] = mha_num_heads
        self.temporal = build_temporal_encoder(
            temporal_type,
            input_dim=gnn_hidden,
            hidden_dim=temporal_hidden,
            dropout=dropout,
            **extra_kwargs,
        )
        self.fc = nn.Linear(self.temporal.output_dim, 1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N, F), adjacency: (N, N), returns: (B, N)."""
        batch_size, steps, num_regions, _ = x.shape

        hidden_per_step = []
        for step in range(steps):
            hidden_per_step.append(self.gnn(x[:, step, :, :], adjacency))
        hidden_seq = torch.stack(hidden_per_step, dim=1)

        hidden_seq = hidden_seq.permute(0, 2, 1, 3).contiguous().view(
            batch_size * num_regions, steps, self.gnn_hidden
        )
        temporal_hidden = self.temporal(hidden_seq)

        return self.fc(temporal_hidden).squeeze(-1).view(batch_size, num_regions)

