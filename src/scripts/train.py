"""
Phase 2 training pipeline for predictive policing.

This script:
1. Builds a raw training CSV from the yearly Chicago archives if needed.
2. Applies the shared DataProcessor pipeline.
3. Trains multiple models and saves reproducible artifacts.

Example:
    python src/scripts/train.py --start-year 2022 --end-year 2024
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import random
import sys
from typing import Callable
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from src.data.dataset_builder import BuildSummary, build_phase2_dataset
from src.data.processor import DataProcessor

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - best-effort fallback
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(exc)

RANDOM_STATE = 42
DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if TORCH_AVAILABLE
    else None
)

MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
PREDICTIONS_DIR = METRICS_DIR / "predictions"
DATA_SUMMARY_PATH = METRICS_DIR / "phase2_data_summary.json"
MODEL_METRICS_PATH = METRICS_DIR / "phase2_model_metrics.csv"
TRIAL_METRICS_PATH = METRICS_DIR / "phase2_model_trials.csv"
FEATURE_COLUMNS_PATH = METRICS_DIR / "phase2_feature_columns.txt"


SKLEARN_MODEL_GRIDS: dict[str, list[dict]] = {
    "logistic_regression": [
        {"C": 0.5, "max_iter": 400},
        {"C": 1.0, "max_iter": 600},
    ],
    "random_forest": [
        {
            "n_estimators": 200,
            "max_depth": 16,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "max_samples": 0.35,
        },
        {
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "max_samples": 0.25,
        },
    ],
    "hist_gradient_boosting": [
        {
            "learning_rate": 0.08,
            "max_depth": 8,
            "max_iter": 200,
            "min_samples_leaf": 100,
            "l2_regularization": 0.0,
        },
        {
            "learning_rate": 0.05,
            "max_depth": 10,
            "max_iter": 300,
            "min_samples_leaf": 80,
            "l2_regularization": 0.1,
        },
    ],
}

MLP_CONFIGS = [
    {
        "hidden_dims": [128, 64, 32],
        "dropout": 0.30,
        "epochs": 25,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 512,
        "patience": 5,
    },
    {
        "hidden_dims": [192, 96, 48],
        "dropout": 0.25,
        "epochs": 25,
        "lr": 8e-4,
        "weight_decay": 1e-4,
        "batch_size": 512,
        "patience": 5,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 2 training pipeline.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "chicago_crime_2022_2024_phase2.csv",
        help="CSV used by the DataProcessor.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "apps" / "dashboard" / "split_data_by_year",
        help="Yearly source archives used when the raw CSV does not exist.",
    )
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--max-rows-per-year",
        type=int,
        default=75000,
        help="Cap per year to keep local runs practical. Use 0 for full-year data.",
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Force rebuilding the raw CSV from the yearly archives.",
    )
    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=1.0,
        help="Negative-to-positive ratio used by DataProcessor.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
            "crime_mlp",
        ],
        choices=[
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
            "crime_mlp",
        ],
        help="Models to train.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def safe_metric(metric_fn: Callable, *args, **kwargs) -> float:
    try:
        return float(metric_fn(*args, **kwargs))
    except ValueError:
        return float("nan")


def evaluate_split(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": safe_metric(accuracy_score, y_true, y_pred),
        "precision": safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "recall": safe_metric(recall_score, y_true, y_pred, zero_division=0),
        "f1": safe_metric(f1_score, y_true, y_pred, zero_division=0),
        "auroc": safe_metric(roc_auc_score, y_true, y_prob),
        "auprc": safe_metric(average_precision_score, y_true, y_prob),
    }


def save_predictions(model_name: str, split_name: str, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "y_true": y_true.astype(int),
            "y_prob": y_prob.astype(float),
            "y_pred": (y_prob >= 0.5).astype(int),
        }
    ).to_csv(PREDICTIONS_DIR / f"{model_name}_{split_name}.csv", index=False)


def save_pickle(model_name: str, model: object) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with (MODELS_DIR / f"{model_name}.pkl").open("wb") as handle:
        pickle.dump(model, handle)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_dataset(args: argparse.Namespace) -> BuildSummary | None:
    max_rows_per_year = None if args.max_rows_per_year == 0 else args.max_rows_per_year
    if args.data_path.exists() and not args.rebuild_data:
        return None
    return build_phase2_dataset(
        source_dir=args.source_dir,
        output_path=args.data_path,
        start_year=args.start_year,
        end_year=args.end_year,
        max_rows_per_year=max_rows_per_year,
        overwrite=True,
    )


def prepare_data(data_path: Path, neg_ratio: float) -> tuple[pd.DataFrame, ...]:
    processor = DataProcessor(str(data_path), neg_ratio=neg_ratio, random_state=RANDOM_STATE)
    processor.load_and_split()

    X_train, y_train = processor.fit_transform_train()
    X_val, y_val = processor.transform(processor.val_idx)
    X_test, y_test = processor.transform(processor.test_idx)

    with FEATURE_COLUMNS_PATH.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(X_train.columns.tolist()))

    summary = {
        "data_path": str(data_path),
        "raw_rows_after_negative_sampling": int(len(processor.raw_data)),
        "positive_rows": int(processor.labels.sum()),
        "negative_rows": int(len(processor.labels) - processor.labels.sum()),
        "overall_positive_rate": float(processor.labels.mean()),
        "active_historical_windows": list(processor.hist_windows.keys()),
        "feature_count": int(X_train.shape[1]),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
        "test_positive_rate": float(y_test.mean()),
    }
    save_json(DATA_SUMMARY_PATH, summary)

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_sklearn_model(model_name: str, params: dict) -> object:
    if model_name == "logistic_regression":
        return LogisticRegression(
            solver="saga",
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            **params,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            **params,
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            **params,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def predict_probabilities(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-decision))
    raise AttributeError("Model does not expose predict_proba or decision_function.")


def run_sklearn_candidates(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[list[dict], list[dict]]:
    trials: list[dict] = []
    metrics_rows: list[dict] = []
    best_payload: tuple[float, object, dict, np.ndarray, np.ndarray] | None = None

    for trial_index, params in enumerate(SKLEARN_MODEL_GRIDS[model_name], start=1):
        model = build_sklearn_model(model_name, params)
        model.fit(X_train, y_train)

        val_prob = predict_probabilities(model, X_val)
        val_metrics = evaluate_split(y_val.to_numpy(), val_prob)
        trials.append(
            {
                "model": model_name,
                "trial": trial_index,
                "split": "val",
                "params": json.dumps(params, sort_keys=True),
                **val_metrics,
            }
        )

        score = val_metrics["auroc"]
        if best_payload is None or np.nan_to_num(score, nan=-1.0) > np.nan_to_num(best_payload[0], nan=-1.0):
            test_prob = predict_probabilities(model, X_test)
            best_payload = (score, model, params, val_prob, test_prob)

    assert best_payload is not None
    _, best_model, best_params, best_val_prob, best_test_prob = best_payload
    save_pickle(model_name, best_model)

    for split_name, y_split, y_prob in [
        ("val", y_val.to_numpy(), best_val_prob),
        ("test", y_test.to_numpy(), best_test_prob),
    ]:
        split_metrics = evaluate_split(y_split, y_prob)
        metrics_rows.append(
            {
                "model": model_name,
                "split": split_name,
                "params": json.dumps(best_params, sort_keys=True),
                **split_metrics,
            }
        )
        save_predictions(model_name, split_name, y_split, y_prob)

    return trials, metrics_rows


if TORCH_AVAILABLE:
    class CrimeDataset(Dataset):
        def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
            self.X = torch.as_tensor(X.to_numpy(dtype=np.float32), dtype=torch.float32)
            self.y = torch.as_tensor(y.to_numpy(dtype=np.float32), dtype=torch.float32)

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, idx: int):
            return self.X[idx], self.y[idx]


    class CrimeMLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                )
                prev_dim = hidden_dim
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(prev_dim, 1)
            self._init_weights()

        def _init_weights(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.backbone(x)).squeeze(1)


    def make_loader(
        X: pd.DataFrame,
        y: pd.Series,
        *,
        batch_size: int,
        shuffle: bool = False,
        weighted: bool = False,
    ) -> DataLoader:
        dataset = CrimeDataset(X, y)
        kwargs = {"batch_size": batch_size, "num_workers": 0, "pin_memory": DEVICE.type == "cuda"}
        if weighted:
            labels = y.to_numpy(dtype=float)
            pos_count = max(labels.sum(), 1.0)
            neg_count = max(len(labels) - pos_count, 1.0)
            weights = torch.as_tensor(
                [1.0 / pos_count if label == 1 else 1.0 / neg_count for label in labels],
                dtype=torch.float32,
            )
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            return DataLoader(dataset, sampler=sampler, **kwargs)
        return DataLoader(dataset, shuffle=shuffle, **kwargs)


    @torch.no_grad()
    def predict_mlp(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        probs, labels = [], []
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(DEVICE, non_blocking=True))
            probs.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(y_batch.numpy())
        return np.asarray(probs, dtype=float), np.asarray(labels, dtype=float)


    def train_single_mlp(
        config: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[nn.Module, list[dict], np.ndarray, dict[str, float]]:
        train_loader = make_loader(X_train, y_train, batch_size=config["batch_size"], weighted=True)
        val_loader = make_loader(X_val, y_val, batch_size=config["batch_size"])

        model = CrimeMLP(X_train.shape[1], config["hidden_dims"], config["dropout"]).to(DEVICE)
        pos_count = max(float(y_train.sum()), 1.0)
        neg_count = max(float(len(y_train) - y_train.sum()), 1.0)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.as_tensor([neg_count / pos_count], device=DEVICE)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        best_state = None
        best_score = -1.0
        patience = 0
        history: list[dict] = []

        for epoch in range(1, config["epochs"] + 1):
            model.train()
            total_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE, non_blocking=True)
                y_batch = y_batch.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.item())

            val_prob, _ = predict_mlp(model, val_loader)
            val_metrics = evaluate_split(y_val.to_numpy(), val_prob)
            avg_loss = total_loss / max(len(train_loader), 1)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_auroc": val_metrics["auroc"],
                    "val_auprc": val_metrics["auprc"],
                }
            )

            score = val_metrics["auroc"]
            if np.nan_to_num(score, nan=-1.0) > np.nan_to_num(best_score, nan=-1.0):
                best_score = score
                patience = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                patience += 1
                if patience >= config["patience"]:
                    break

        assert best_state is not None
        model.load_state_dict(best_state)
        val_prob, _ = predict_mlp(model, val_loader)
        val_metrics = evaluate_split(y_val.to_numpy(), val_prob)
        return model, history, val_prob, val_metrics


def run_mlp_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[list[dict], list[dict]]:
    trials: list[dict] = []
    metrics_rows: list[dict] = []

    if not TORCH_AVAILABLE:
        trials.append(
            {
                "model": "crime_mlp",
                "trial": 0,
                "split": "skipped",
                "params": "{}",
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "auroc": np.nan,
                "auprc": np.nan,
                "note": TORCH_IMPORT_ERROR,
            }
        )
        return trials, metrics_rows

    best_payload: tuple[float, nn.Module, dict, np.ndarray, list[dict]] | None = None

    for trial_index, config in enumerate(MLP_CONFIGS, start=1):
        model, history, val_prob, val_metrics = train_single_mlp(
            config, X_train, y_train, X_val, y_val
        )
        trials.append(
            {
                "model": "crime_mlp",
                "trial": trial_index,
                "split": "val",
                "params": json.dumps(config, sort_keys=True),
                **val_metrics,
            }
        )
        score = val_metrics["auroc"]
        if best_payload is None or np.nan_to_num(score, nan=-1.0) > np.nan_to_num(best_payload[0], nan=-1.0):
            best_payload = (score, model, config, val_prob, history)

    assert best_payload is not None
    _, best_model, best_config, best_val_prob, best_history = best_payload

    test_loader = make_loader(X_test, y_test, batch_size=best_config["batch_size"])
    best_test_prob, _ = predict_mlp(best_model, test_loader)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_model.state_dict(),
            "config": best_config,
            "input_dim": X_train.shape[1],
            "feature_columns": X_train.columns.tolist(),
        },
        MODELS_DIR / "crime_mlp.pt",
    )
    pd.DataFrame(best_history).to_csv(METRICS_DIR / "crime_mlp_history.csv", index=False)

    for split_name, y_split, y_prob in [
        ("val", y_val.to_numpy(), best_val_prob),
        ("test", y_test.to_numpy(), best_test_prob),
    ]:
        split_metrics = evaluate_split(y_split, y_prob)
        metrics_rows.append(
            {
                "model": "crime_mlp",
                "split": split_name,
                "params": json.dumps(best_config, sort_keys=True),
                **split_metrics,
            }
        )
        save_predictions("crime_mlp", split_name, y_split, y_prob)

    return trials, metrics_rows


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_STATE)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    build_summary = ensure_dataset(args)
    if build_summary is not None:
        save_json(METRICS_DIR / "phase2_dataset_build_summary.json", build_summary.to_dict())

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(args.data_path, args.neg_ratio)

    trial_rows: list[dict] = []
    metric_rows: list[dict] = []

    for model_name in args.models:
        if model_name == "crime_mlp":
            trials, metrics = run_mlp_candidates(X_train, y_train, X_val, y_val, X_test, y_test)
        else:
            trials, metrics = run_sklearn_candidates(
                model_name, X_train, y_train, X_val, y_val, X_test, y_test
            )
        trial_rows.extend(trials)
        metric_rows.extend(metrics)

    pd.DataFrame(trial_rows).to_csv(TRIAL_METRICS_PATH, index=False)
    pd.DataFrame(metric_rows).sort_values(["split", "auroc"], ascending=[True, False]).to_csv(
        MODEL_METRICS_PATH, index=False
    )

    manifest = {
        "data_path": str(args.data_path),
        "models_trained": args.models,
        "dataset_build_summary_present": build_summary is not None,
        "metrics_path": str(MODEL_METRICS_PATH),
        "trial_metrics_path": str(TRIAL_METRICS_PATH),
    }
    save_json(METRICS_DIR / "phase2_training_manifest.json", manifest)

    print(pd.DataFrame(metric_rows).sort_values(["split", "auroc"], ascending=[True, False]))


if __name__ == "__main__":
    main()
