"""
Compute spatial and temporal evaluation summaries for saved model predictions.

This script reconstructs the aligned validation/test metadata after preprocessing,
joins it with the saved prediction files, and exports compact spatial-temporal
metrics that can be cited in the report.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.processor import DataProcessor
from src.data.split_strategy import build_year_holdout_splits

RANDOM_STATE = 42
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "chicago_crime_district_hour_2015_2025_phase2.csv"
PREDICTIONS_DIR = PROJECT_ROOT / "artifacts" / "metrics" / "predictions"
OUTPUT_CSV = PROJECT_ROOT / "artifacts" / "metrics" / "phase2_spatiotemporal_metrics.csv"
OUTPUT_JSON = PROJECT_ROOT / "artifacts" / "metrics" / "phase2_spatiotemporal_summary.json"
TRAIN_START_YEAR = 2015
TRAIN_END_YEAR = 2024
HOLDOUT_YEAR = 2025
HOLDOUT_VAL_FRACTION = 0.5

MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "hist_gradient_boosting",
]
SPLITS = ["val", "test"]


def aligned_metadata(processor: DataProcessor, split_name: str) -> pd.DataFrame:
    split_idx = getattr(processor, f"{split_name}_idx")
    raw = processor.raw_data.iloc[split_idx].copy().reset_index(drop=True)
    labels = processor.labels.iloc[split_idx].reset_index(drop=True)
    _, mask, _ = processor._process(raw, labels, is_train=False)

    aligned = raw.loc[mask].reset_index(drop=True)
    aligned["y_true"] = labels.loc[mask].reset_index(drop=True).astype(int)
    aligned["district"] = aligned["District"].fillna(-1).astype(int)
    aligned["hour"] = pd.to_datetime(aligned["Date"]).dt.hour
    aligned["day_of_week"] = pd.to_datetime(aligned["Date"]).dt.dayofweek
    return aligned


def safe_corr(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or left.nunique() <= 1 or right.nunique() <= 1:
        return float("nan")
    return float(left.corr(right))


def overlap_score(actual_rank: list[int], predicted_rank: list[int], top_k: int) -> float:
    actual_top = set(actual_rank[:top_k])
    pred_top = set(predicted_rank[:top_k])
    if top_k <= 0:
        return float("nan")
    return len(actual_top & pred_top) / top_k


def spatial_metrics(df: pd.DataFrame) -> dict[str, float]:
    grouped = (
        df.groupby("district", as_index=False)
        .agg(
            actual_count=("y_true", "sum"),
            average_predicted_risk=("y_prob", "mean"),
        )
        .sort_values("district")
        .reset_index(drop=True)
    )

    actual_rank = (
        grouped.sort_values(["actual_count", "district"], ascending=[False, True])["district"].tolist()
    )
    predicted_rank = (
        grouped.sort_values(
            ["average_predicted_risk", "district"], ascending=[False, True]
        )["district"].tolist()
    )

    return {
        "district_correlation": safe_corr(
            grouped["actual_count"], grouped["average_predicted_risk"]
        ),
        "top_5_overlap": overlap_score(actual_rank, predicted_rank, top_k=5),
        "top_10_overlap": overlap_score(actual_rank, predicted_rank, top_k=10),
    }


def temporal_metrics(df: pd.DataFrame) -> dict[str, float]:
    hourly = (
        df.groupby("hour", as_index=False)
        .agg(
            actual_rate=("y_true", "mean"),
            predicted_risk=("y_prob", "mean"),
        )
        .sort_values("hour")
        .reset_index(drop=True)
    )
    weekday = (
        df.groupby("day_of_week", as_index=False)
        .agg(
            actual_rate=("y_true", "mean"),
            predicted_risk=("y_prob", "mean"),
        )
        .sort_values("day_of_week")
        .reset_index(drop=True)
    )

    return {
        "hourly_correlation": safe_corr(hourly["actual_rate"], hourly["predicted_risk"]),
        "day_of_week_correlation": safe_corr(
            weekday["actual_rate"], weekday["predicted_risk"]
        ),
    }


def main() -> None:
    processor = DataProcessor(str(DATA_PATH), neg_ratio=1.0, random_state=RANDOM_STATE)
    processor.load_and_split()
    train_idx, val_idx, test_idx, _ = build_year_holdout_splits(
        processor.raw_data,
        train_start_year=TRAIN_START_YEAR,
        train_end_year=TRAIN_END_YEAR,
        holdout_year=HOLDOUT_YEAR,
        holdout_val_fraction=HOLDOUT_VAL_FRACTION,
    )
    processor.set_split_indices(
        train_idx,
        val_idx,
        test_idx,
        split_label=f"train {TRAIN_START_YEAR}-{TRAIN_END_YEAR}, holdout {HOLDOUT_YEAR}",
    )
    processor.fit_transform_train()

    metadata_by_split = {
        split_name: aligned_metadata(processor, split_name)
        for split_name in SPLITS
    }

    rows: list[dict] = []
    summary: dict[str, dict] = {}

    for model_name in MODELS:
        summary[model_name] = {}
        for split_name in SPLITS:
            pred_path = PREDICTIONS_DIR / f"{model_name}_{split_name}.csv"
            if not pred_path.exists():
                continue

            pred_df = pd.read_csv(pred_path)
            joined = metadata_by_split[split_name].copy()
            joined["y_prob"] = pred_df["y_prob"].astype(float).to_numpy()
            joined["y_pred"] = pred_df["y_pred"].astype(int).to_numpy()

            metrics = {
                **spatial_metrics(joined),
                **temporal_metrics(joined),
            }
            row = {"model": model_name, "split": split_name, **metrics}
            rows.append(row)
            summary[model_name][split_name] = metrics

    result_df = pd.DataFrame(rows).sort_values(["split", "model"]).reset_index(drop=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(result_df.to_string(index=False))
    print(f"\nSaved spatial-temporal metrics to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
