"""Adjacency-matrix utilities for the integrated Jiadong STGNN module."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.experimental.jiadong_stgnn.config import KNN_K, REGION_IDS


def compute_region_centroids(
    df: pd.DataFrame,
    region_ids: List[int] = REGION_IDS,
) -> np.ndarray:
    """Compute one centroid per Chicago community area."""
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    df["region_id"] = df["Community Area"].astype(int)
    grouped = df.groupby("region_id")[["Latitude", "Longitude"]].mean()

    centroids = np.zeros((len(region_ids), 2))
    for idx, region_id in enumerate(region_ids):
        if region_id in grouped.index:
            centroids[idx] = grouped.loc[region_id].values
        else:
            centroids[idx] = grouped.mean().values
            print(f"[warn] region {region_id} has no lat/lon data, using global mean")
    return centroids


def build_knn_adjacency(centroids: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """Build a binary KNN adjacency matrix from region centroids."""
    count = centroids.shape[0]
    dist_matrix = cdist(centroids, centroids, metric="euclidean")
    adjacency = np.zeros((count, count), dtype=np.float32)
    for idx in range(count):
        neighbors = np.argsort(dist_matrix[idx])[1 : k + 1]
        adjacency[idx, neighbors] = 1.0
    return adjacency


def symmetrize(adjacency: np.ndarray) -> np.ndarray:
    return np.maximum(adjacency, adjacency.T)


def add_self_loops(adjacency: np.ndarray) -> np.ndarray:
    return adjacency + np.eye(adjacency.shape[0], dtype=adjacency.dtype)


def gcn_normalize(adjacency: np.ndarray) -> np.ndarray:
    degree = np.sum(adjacency, axis=1)
    degree_inv_sqrt = np.where(degree > 0, np.power(degree, -0.5), 0.0)
    degree_matrix = np.diag(degree_inv_sqrt)
    return degree_matrix @ adjacency @ degree_matrix


def build_adjacency(
    df: pd.DataFrame,
    k: int = KNN_K,
    region_ids: List[int] = REGION_IDS,
) -> np.ndarray:
    """Full adjacency-construction pipeline."""
    centroids = compute_region_centroids(df, region_ids)
    adjacency = build_knn_adjacency(centroids, k)
    adjacency = symmetrize(adjacency)
    adjacency = add_self_loops(adjacency)
    adjacency = gcn_normalize(adjacency)
    print(f"[adjacency] shape={adjacency.shape}, non-zero={np.count_nonzero(adjacency)}")
    return adjacency


def save_adjacency(adjacency: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, adjacency)
    print(f"[adjacency] saved -> {path}")


def load_adjacency(path: str | Path) -> np.ndarray:
    return np.load(path)

