"""
Compute feature-importance artifacts for Phase 2 Part 4.

This script produces:
- model-level feature importance CSVs
- top-feature bar plots
- a grouped feature-importance summary

It is intended to strengthen the `Feature Importance Analysis` section
for Milestone 2 by providing concrete outputs instead of text-only claims.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / ".cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.train import prepare_data

RANDOM_STATE = 42
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "chicago_crime_2022_2024_phase2.csv"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "metrics" / "feature_importance"


def feature_group(feature_name: str) -> str:
    if feature_name.startswith("crimes_last_") or feature_name == "crime_density_500m":
        return "historical"
    if feature_name in {
        "hour",
        "day_of_week",
        "month",
        "day_of_year",
        "week_of_year",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_night",
        "is_rush_hour",
    }:
        return "temporal"
    if feature_name.endswith("_missing"):
        return "missing_indicator"
    return "spatial"


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_importance_df(feature_names: list[str], values: np.ndarray, model_name: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values.astype(float),
            "model": model_name,
        }
    )
    df["importance_abs"] = df["importance"].abs()
    df["feature_group"] = df["feature"].map(feature_group)
    return df.sort_values("importance_abs", ascending=False).reset_index(drop=True)


def save_bar_plot(df: pd.DataFrame, output_path: Path, title: str, top_n: int = 15) -> None:
    plot_df = df.head(top_n).copy().iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df["importance_abs"], color="#3b82f6")
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def summarize_groups(df: pd.DataFrame) -> pd.DataFrame:
    group_df = (
        df.groupby("feature_group", as_index=False)["importance_abs"]
        .sum()
        .sort_values("importance_abs", ascending=False)
        .reset_index(drop=True)
    )
    total = group_df["importance_abs"].sum()
    group_df["importance_share"] = (
        group_df["importance_abs"] / total if total > 0 else 0.0
    )
    return group_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _, _, _, _, X_test, y_test = prepare_data(DATA_PATH, neg_ratio=1.0)
    feature_names = X_test.columns.tolist()

    logistic_model = load_pickle(MODELS_DIR / "logistic_regression.pkl")
    random_forest_model = load_pickle(MODELS_DIR / "random_forest.pkl")
    hist_gb_model = load_pickle(MODELS_DIR / "hist_gradient_boosting.pkl")

    logistic_df = build_importance_df(
        feature_names,
        np.abs(logistic_model.coef_[0]),
        "logistic_regression",
    )
    random_forest_df = build_importance_df(
        feature_names,
        random_forest_model.feature_importances_,
        "random_forest",
    )

    permutation = permutation_importance(
        hist_gb_model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="average_precision",
        n_jobs=1,
    )
    hist_gb_df = build_importance_df(
        feature_names,
        permutation.importances_mean,
        "hist_gradient_boosting",
    )
    hist_gb_df["importance_std"] = permutation.importances_std

    outputs = {
        "logistic_regression": logistic_df,
        "random_forest": random_forest_df,
        "hist_gradient_boosting": hist_gb_df,
    }

    for model_name, df in outputs.items():
        df.to_csv(OUTPUT_DIR / f"{model_name}_feature_importance.csv", index=False)
        save_bar_plot(
            df,
            OUTPUT_DIR / f"{model_name}_feature_importance_top15.png",
            title=f"Top 15 Feature Importances ({model_name})",
        )

    hist_group_df = summarize_groups(hist_gb_df)
    hist_group_df["model"] = "hist_gradient_boosting"
    hist_group_df.to_csv(OUTPUT_DIR / "hist_gradient_boosting_feature_groups.csv", index=False)

    summary_payload = {
        "best_model": "hist_gradient_boosting",
        "scoring_method": "permutation importance with average_precision on test split",
        "top_features": hist_gb_df.head(10)[["feature", "importance", "importance_std"]].to_dict(orient="records"),
        "feature_groups": hist_group_df[["feature_group", "importance_abs", "importance_share"]].to_dict(orient="records"),
    }
    (OUTPUT_DIR / "feature_importance_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    print("Saved feature-importance artifacts to:", OUTPUT_DIR)
    print("\nTop 10 features for HistGradientBoosting:")
    print(hist_gb_df.head(10).to_string(index=False))
    print("\nGrouped importance summary:")
    print(hist_group_df.to_string(index=False))


if __name__ == "__main__":
    main()
