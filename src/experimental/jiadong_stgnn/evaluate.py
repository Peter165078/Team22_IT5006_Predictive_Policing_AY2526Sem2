"""Evaluation utilities for the integrated Jiadong STGNN module."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.experimental.jiadong_stgnn.config import DEVICE, TARGET_IDX


def inverse_scale_target(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Inverse-transform the standardized target column back to crime-count space."""
    mean = scaler.mean_[TARGET_IDX]
    std = scaler.scale_[TARGET_IDX]
    return values * std + mean


def compute_mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def compute_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def compute_smape(pred: np.ndarray, true: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (0-200%)."""
    denom = np.abs(true) + np.abs(pred)
    mask = denom > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(2.0 * np.abs(true[mask] - pred[mask]) / denom[mask]) * 100)


def per_region_mae(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred - true), axis=0)


def per_day_mae(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred - true), axis=1)


def evaluate(
    model,
    loader,
    adj,
    scaler: StandardScaler,
    device: str = DEVICE,
) -> Dict[str, object]:
    """Run evaluation and report both scalar and sliced metrics."""
    import torch

    model = model.to(device)
    model.eval()
    adj = adj.to(device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_hat = model(x_batch, adj)
            all_preds.append(y_hat.cpu().numpy())
            all_labels.append(y_batch.numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    preds_orig = inverse_scale_target(preds, scaler)
    labels_orig = inverse_scale_target(labels, scaler)

    return {
        "MAE": compute_mae(preds_orig, labels_orig),
        "RMSE": compute_rmse(preds_orig, labels_orig),
        "SMAPE": compute_smape(preds_orig, labels_orig),
        "per_region_mae": per_region_mae(preds_orig, labels_orig),
        "per_day_mae": per_day_mae(preds_orig, labels_orig),
        "preds": preds_orig,
        "labels": labels_orig,
    }


def print_metrics(metrics: Dict[str, object], split_name: str = "Test") -> None:
    print(f"\n{'-' * 40}")
    print(f"  {split_name} Metrics (original scale)")
    print(f"{'-' * 40}")
    for key in ("MAE", "RMSE", "SMAPE"):
        print(f"  {key:>10s}: {metrics[key]:.4f}")
    print(f"{'-' * 40}\n")

