"""Dataset helpers for the integrated Jiadong STGNN module."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.experimental.jiadong_stgnn.config import BATCH_SIZE


class CrimeDataset(Dataset):
    """Spatiotemporal crime prediction dataset."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def create_dataloader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
) -> DataLoader:
    """Wrap numpy arrays into a DataLoader."""
    dataset = CrimeDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

