"""Preprocessing pipeline for the integrated Jiadong STGNN module."""
from __future__ import annotations

import pickle
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.experimental.jiadong_stgnn.config import (
    COUNT_FEATURES,
    CRIME_TYPE_MAP,
    FEATURE_COLS,
    REGION_IDS,
    WINDOW_SIZE,
)

REQUIRED_COLS = [
    "ID",
    "Date",
    "Community Area",
    "Primary Type",
    "Latitude",
    "Longitude",
]


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Read the raw Chicago crime CSV."""
    df = pd.read_csv(path)
    print(f"[load] Raw records: {len(df)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the minimum cleaning used in Jiadong's spatiotemporal pipeline."""
    df = df[REQUIRED_COLS].copy()
    before = len(df)
    df.drop_duplicates(subset=["ID"], inplace=True)
    deduped = len(df)
    df.dropna(subset=["Date", "Community Area"], inplace=True)
    after = len(df)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[clean] {before} -> dedup {deduped} -> drop_na {after}")
    return df


def build_daily_features(
    df: pd.DataFrame,
    crime_type_map: Dict[str, str] = CRIME_TYPE_MAP,
) -> pd.DataFrame:
    """Aggregate raw records into daily region-level feature rows."""
    df = df.copy()
    df["date"] = df["Date"].dt.date
    df["region_id"] = df["Community Area"].astype(int)

    for raw_label, col_name in crime_type_map.items():
        df[col_name] = (df["Primary Type"] == raw_label).astype(int)

    agg_dict = {"ID": "count"}
    for col_name in crime_type_map.values():
        agg_dict[col_name] = "sum"

    agg = df.groupby(["region_id", "date"]).agg(agg_dict).reset_index()
    agg.rename(columns={"ID": "crime_count"}, inplace=True)
    return agg


def build_full_grid(
    agg_df: pd.DataFrame,
    region_ids: List[int],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Create a complete region-date grid and fill missing counts with zeros."""
    all_dates = pd.date_range(start_date, end_date, freq="D").date
    idx = pd.MultiIndex.from_product([region_ids, all_dates], names=["region_id", "date"])
    grid = pd.DataFrame(index=idx).reset_index()

    agg_df = agg_df.copy()
    agg_df["region_id"] = agg_df["region_id"].astype(int)
    agg_df["date"] = pd.to_datetime(agg_df["date"]).dt.date

    grid = grid.merge(agg_df, on=["region_id", "date"], how="left")

    for col in COUNT_FEATURES:
        grid[col] = grid[col].fillna(0).astype(float)

    dt_series = pd.to_datetime(grid["date"])
    grid["day_of_week"] = dt_series.dt.dayofweek.astype(float)
    grid["is_weekend"] = (grid["day_of_week"] >= 5).astype(float)
    grid["month"] = dt_series.dt.month.astype(float)

    grid.sort_values(["date", "region_id"], inplace=True)
    grid.reset_index(drop=True, inplace=True)
    print(f"[grid] {len(region_ids)} regions x {len(all_dates)} dates = {len(grid)} rows")
    return grid


def fit_scaler(grid_df: pd.DataFrame, feature_cols: List[str] = FEATURE_COLS) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(grid_df[feature_cols].values)
    print(f"[scaler] fit on {len(grid_df)} rows, {len(feature_cols)} features")
    return scaler


def apply_scaler(
    grid_df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: List[str] = FEATURE_COLS,
) -> pd.DataFrame:
    grid_df = grid_df.copy()
    grid_df[feature_cols] = scaler.transform(grid_df[feature_cols].values)
    return grid_df


def save_scaler(scaler: StandardScaler, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(scaler, handle)
    print(f"[scaler] saved -> {path}")


def load_scaler(path: str | Path) -> StandardScaler:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def build_samples(
    grid_df: pd.DataFrame,
    region_ids: List[int],
    feature_cols: List[str] = FEATURE_COLS,
    window_size: int = WINDOW_SIZE,
    target_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn the complete region-date grid into sliding-window samples."""
    num_regions = len(region_ids)
    dates = sorted(grid_df["date"].unique())
    num_dates = len(dates)
    values = grid_df.sort_values(["date", "region_id"])[feature_cols].values
    data_cube = values.reshape(num_dates, num_regions, len(feature_cols))

    x_list, y_list = [], []
    for idx in range(window_size, num_dates):
        x_list.append(data_cube[idx - window_size : idx])
        y_list.append(data_cube[idx, :, target_idx])

    x = np.array(x_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"[samples] X={x.shape}  Y={y.shape}")
    return x, y


def preprocess_split(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    scaler: Optional[StandardScaler] = None,
    fit_mode: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """End-to-end preprocessing for one temporal split."""
    df = clean_data(df)
    agg = build_daily_features(df)
    grid = build_full_grid(
        agg,
        region_ids=REGION_IDS,
        start_date=date.fromisoformat(start_date),
        end_date=date.fromisoformat(end_date),
    )

    if fit_mode:
        scaler = fit_scaler(grid)
    elif scaler is None:
        raise ValueError("A fitted scaler is required when fit_mode=False.")

    scaled_grid = apply_scaler(grid, scaler)
    x, y = build_samples(scaled_grid, region_ids=REGION_IDS)
    return x, y, scaler

