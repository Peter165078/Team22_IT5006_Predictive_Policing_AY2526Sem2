"""
Run external generalization evaluation on prepared NIBRS county-hour datasets.

This script keeps the Chicago training pipeline intact:
- fit preprocessing statistics on the Chicago train split
- load the saved Chicago model artifact
- transform NIBRS county-hour data using the Chicago-fitted schema/statistics
- evaluate on the chosen external test year
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.processor import DataProcessor
from src.data.split_strategy import build_year_holdout_splits

RANDOM_STATE = 42
CHICAGO_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "chicago_crime_district_hour_2015_2025_phase2.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "hist_gradient_boosting.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "phase2_feature_columns.txt"
DEFAULT_NIBRS_PATHS = [
    PROJECT_ROOT / "data" / "raw" / "nibrs_county_hour_tx_2023_2024.csv",
    PROJECT_ROOT / "data" / "raw" / "nibrs_county_hour_co_2023_2024.csv",
]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "metrics" / "nibrs_generalization"

TRAIN_START_YEAR = 2015
TRAIN_END_YEAR = 2024
HOLDOUT_YEAR = 2025
HOLDOUT_VAL_FRACTION = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the saved Chicago model on NIBRS data.")
    parser.add_argument(
        "--nibrs-paths",
        nargs="+",
        type=Path,
        default=DEFAULT_NIBRS_PATHS,
        help="Prepared NIBRS county-hour CSV paths.",
    )
    parser.add_argument(
        "--eval-year",
        type=int,
        default=2024,
        help="External test year to score within each prepared NIBRS dataset.",
    )
    return parser.parse_args()


def safe_metric(metric_fn, *args, **kwargs) -> float:
    try:
        return float(metric_fn(*args, **kwargs))
    except ValueError:
        return float("nan")


def overall_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "rows": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "accuracy": safe_metric(accuracy_score, y_true, y_pred),
        "precision": safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "recall": safe_metric(recall_score, y_true, y_pred, zero_division=0),
        "f1": safe_metric(f1_score, y_true, y_pred, zero_division=0),
        "auroc": safe_metric(roc_auc_score, y_true, y_prob),
        "auprc": safe_metric(average_precision_score, y_true, y_prob),
    }


def safe_corr(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or left.nunique() <= 1 or right.nunique() <= 1:
        return float("nan")
    return float(left.corr(right))


def county_hour_alignment(meta: pd.DataFrame, y_prob: np.ndarray) -> dict[str, float]:
    df = meta.copy()
    df["y_prob"] = y_prob
    county = (
        df.groupby("District", as_index=False)
        .agg(actual_rate=("target", "mean"), predicted_rate=("y_prob", "mean"))
        .sort_values("District")
        .reset_index(drop=True)
    )
    hourly = (
        df.groupby("hour", as_index=False)
        .agg(actual_rate=("target", "mean"), predicted_rate=("y_prob", "mean"))
        .sort_values("hour")
        .reset_index(drop=True)
    )
    return {
        "county_correlation": safe_corr(county["actual_rate"], county["predicted_rate"]),
        "hourly_correlation": safe_corr(hourly["actual_rate"], hourly["predicted_rate"]),
    }


def load_feature_columns() -> list[str]:
    if FEATURE_COLUMNS_PATH.exists():
        return [
            line.strip()
            for line in FEATURE_COLUMNS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    raise FileNotFoundError(f"Missing feature column manifest: {FEATURE_COLUMNS_PATH}")


def fit_chicago_processor() -> tuple[DataProcessor, list[str]]:
    processor = DataProcessor(str(CHICAGO_DATA_PATH), neg_ratio=1.0, random_state=RANDOM_STATE)
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
    X_train, _ = processor.fit_transform_train()
    feature_columns = load_feature_columns() if FEATURE_COLUMNS_PATH.exists() else X_train.columns.tolist()
    return processor, feature_columns


def align_feature_schema(X: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    aligned = X.copy()
    missing = [col for col in feature_columns if col not in aligned.columns]
    for col in missing:
        aligned[col] = 0.0
    aligned = aligned[feature_columns]
    return aligned


def evaluate_external_dataset(
    data_path: Path,
    *,
    eval_year: int,
    chicago_fit: DataProcessor,
    feature_columns: list[str],
    model,
) -> dict:
    processor = DataProcessor(
        str(data_path),
        neg_ratio=1.0,
        random_state=RANDOM_STATE,
        spatial_bounds_mode="passthrough",
    )
    processor.load_and_split()
    processor.fit_dict = copy.deepcopy(chicago_fit.fit_dict)

    raw = processor.raw_data.copy()
    labels = processor.labels.copy()
    raw["Date"] = pd.to_datetime(raw["Date"])
    year_mask = raw["Date"].dt.year == eval_year
    raw_eval = raw.loc[year_mask].reset_index(drop=True)
    labels_eval = labels.loc[year_mask].reset_index(drop=True)

    X_eval, mask, _ = processor._process(raw_eval, labels_eval, is_train=False)
    y_eval = labels_eval.loc[mask].astype(int).reset_index(drop=True)
    meta = raw_eval.loc[mask].reset_index(drop=True)
    meta["target"] = y_eval
    meta["hour"] = pd.to_datetime(meta["Date"]).dt.hour

    X_eval = align_feature_schema(X_eval, feature_columns)
    y_prob = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "dataset": data_path.name,
        "eval_year": int(eval_year),
        **overall_metrics(y_eval.to_numpy(), y_prob),
        **county_hour_alignment(meta, y_prob),
        "distinct_counties": int(meta["District"].nunique()),
        "date_min": str(pd.to_datetime(meta["Date"]).min()),
        "date_max": str(pd.to_datetime(meta["Date"]).max()),
    }
    return metrics


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chicago_processor, feature_columns = fit_chicago_processor()
    with MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)

    rows: list[dict] = []
    for data_path in args.nibrs_paths:
        if not data_path.exists():
            raise FileNotFoundError(f"Missing prepared NIBRS dataset: {data_path}")
        metrics = evaluate_external_dataset(
            data_path,
            eval_year=args.eval_year,
            chicago_fit=chicago_processor,
            feature_columns=feature_columns,
            model=model,
        )
        rows.append(metrics)

    result_df = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    csv_path = OUTPUT_DIR / f"nibrs_generalization_metrics_{args.eval_year}.csv"
    json_path = OUTPUT_DIR / f"nibrs_generalization_metrics_{args.eval_year}.json"
    result_df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )

    print(result_df.to_string(index=False))
    print(f"\nSaved NIBRS generalization metrics to: {csv_path}")


if __name__ == "__main__":
    main()
