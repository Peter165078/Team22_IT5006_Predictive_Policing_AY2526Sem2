"""Experiment helpers for the integrated Jiadong STGNN module."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.experimental.jiadong_stgnn.config import (
    ADJACENCY_PATH,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    RAW_DATA_PATH,
    RESULTS_DIR,
    SCALER_PATH,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)
from src.experimental.jiadong_stgnn.utils.adjacency import build_adjacency, load_adjacency, save_adjacency
from src.experimental.jiadong_stgnn.utils.data_pipeline import (
    clean_data,
    load_raw_data,
    load_scaler,
    preprocess_split,
    save_scaler,
)
from src.experimental.jiadong_stgnn.utils.dataset import create_dataloader


def prepare_data(force_rebuild: bool = False):
    """Load or rebuild preprocessed data plus adjacency matrix."""
    scaler_exists = Path(SCALER_PATH).exists()
    adj_exists = Path(ADJACENCY_PATH).exists()

    if not force_rebuild and scaler_exists and adj_exists:
        print("[data] Loading cached scaler and adjacency ...")
        scaler = load_scaler(SCALER_PATH)
        adj_np = load_adjacency(ADJACENCY_PATH)
    else:
        print("[data] Building from scratch ...")
        scaler = None
        adj_np = None

    raw_df = load_raw_data(RAW_DATA_PATH)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"], format="mixed")

    day_values = raw_df["Date"].dt.date
    train_raw = raw_df[day_values <= date.fromisoformat(TRAIN_END)].copy()
    val_raw = raw_df[
        (day_values >= date.fromisoformat(VAL_START)) & (day_values <= date.fromisoformat(VAL_END))
    ].copy()
    test_raw = raw_df[(day_values >= date.fromisoformat(TEST_START)) & (day_values <= date.fromisoformat(TEST_END))].copy()
    print(f"[split] train={len(train_raw)}  val={len(val_raw)}  test={len(test_raw)}")

    if adj_np is None:
        train_cleaned = clean_data(train_raw.copy())
        adj_np = build_adjacency(train_cleaned)
        save_adjacency(adj_np, ADJACENCY_PATH)

    if scaler is None:
        x_train, y_train, scaler = preprocess_split(train_raw, TRAIN_START, TRAIN_END, fit_mode=True)
        save_scaler(scaler, SCALER_PATH)
    else:
        x_train, y_train, _ = preprocess_split(train_raw, TRAIN_START, TRAIN_END, scaler=scaler, fit_mode=False)

    x_val, y_val, _ = preprocess_split(val_raw, VAL_START, VAL_END, scaler=scaler, fit_mode=False)
    x_test, y_test, _ = preprocess_split(test_raw, TEST_START, TEST_END, scaler=scaler, fit_mode=False)

    train_loader = create_dataloader(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = create_dataloader(x_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[shapes] X_train={x_train.shape}  X_val={x_val.shape}  X_test={x_test.shape}")
    adj_tensor = torch.tensor(adj_np, dtype=torch.float32)
    return train_loader, val_loader, test_loader, adj_tensor, scaler


def save_run_results(
    run_name: str,
    history: dict,
    val_metrics: dict,
    test_metrics: dict,
) -> Path:
    """Save scalar metrics plus sliced arrays for a completed run."""
    out_dir = Path(RESULTS_DIR) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    scalars = {
        "val": {key: value for key, value in val_metrics.items() if isinstance(value, (int, float))},
        "test": {key: value for key, value in test_metrics.items() if isinstance(value, (int, float))},
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(scalars, handle, indent=2)

    np.save(out_dir / "val_per_region_mae.npy", val_metrics["per_region_mae"])
    np.save(out_dir / "val_per_day_mae.npy", val_metrics["per_day_mae"])
    np.save(out_dir / "test_per_region_mae.npy", test_metrics["per_region_mae"])
    np.save(out_dir / "test_per_day_mae.npy", test_metrics["per_day_mae"])

    with open(out_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle)

    print(f"[results] saved -> {out_dir}")
    return out_dir

