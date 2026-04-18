"""Training loop for the integrated Jiadong STGNN experiments."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.experimental.jiadong_stgnn.config import (
    DEVICE,
    GRAD_CLIP_MAX_NORM,
    HUBER_DELTA,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    WEIGHT_DECAY,
)

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    adj: torch.Tensor,
    *,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = PATIENCE,
    device: str = DEVICE,
    save_path: str | Path = "artifacts/models/jiadong_stgnn/best_model.pt",
    use_wandb: bool = False,
) -> dict:
    """Train a model with Huber loss, clipping, and early stopping."""
    model = model.to(device)
    adj = adj.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_hat = model(x_batch, adj)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_hat = model(x_batch, adj)
                loss = criterion(y_hat, y_batch)
                val_losses.append(loss.item())

        epoch_train = float(np.mean(train_losses))
        epoch_val = float(np.mean(val_losses))
        history["train_loss"].append(epoch_train)
        history["val_loss"].append(epoch_val)

        if use_wandb and HAS_WANDB:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": epoch_train,
                    "val_loss": epoch_val,
                }
            )

        improved = ""
        if epoch_val < best_val_loss:
            best_val_loss = epoch_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            improved = "  * saved"
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:3d}/{num_epochs}  "
            f"train={epoch_train:.4f}  val={epoch_val:.4f}{improved}"
        )

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"Best val_loss = {best_val_loss:.4f}")
    return history

