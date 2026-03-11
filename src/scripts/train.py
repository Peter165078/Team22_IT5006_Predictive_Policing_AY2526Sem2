"""
src/scripts/train.py

Demonstration script: how to use DataProcessor for the Chicago crime
prediction task, followed by a simple baseline model training.

Pipeline
--------
    DataProcessor.load_and_split()
        → fit_transform_train()         # fit + transform train
        → transform(val_idx)            # transform val  (no refit)
        → transform(test_idx)           # transform test (no refit)
        → [optionally] fit_transform_fold() for CV

Run
---
    uv run python src/scripts/train.py
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

from src.data.processor import DataProcessor

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

DATA_PATH    = PROJECT_ROOT / "data" / "raw" / "Crimes_-_2001_to_Present_20260216.csv"
RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, X: pd.DataFrame, y: pd.Series, tag: str) -> dict:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    auroc  = roc_auc_score(y, y_prob)
    auprc  = average_precision_score(y, y_prob)

    print(f"\n{'='*55}")
    print(f"  [{tag}]")
    print(f"{'='*55}")
    print(f"  AUROC : {auroc:.4f}  |  AUPRC : {auprc:.4f}\n")
    print(classification_report(
        y, y_pred, target_names=["No Crime (0)", "Crime (1)"]
    ))
    return {"tag": tag, "auroc": auroc, "auprc": auprc}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Step 1: Build DataProcessor and split ────────────────────────────────
    print("\n" + "="*55)
    print("  Step 1: Load data & chronological split")
    print("="*55)
    proc = DataProcessor(
        data_path=str(DATA_PATH),
        neg_ratio=1.0,          # 1:1 positive:negative
        random_state=RANDOM_STATE,
    )
    proc.load_and_split()

    # ── Step 2: Fit & transform (train only) ─────────────────────────────────
    print("\n" + "="*55)
    print("  Step 2: Fit & transform train")
    print("="*55)
    X_train = proc.fit_transform_train()
    y_train = proc.labels.iloc[proc.train_idx].reset_index(drop=True)

    print(f"  X_train : {X_train.shape}  |  pos rate {y_train.mean():.3f}")

    # ── Step 3: Transform val and test (using train-fitted statistics) ────────
    print("\n" + "="*55)
    print("  Step 3: Transform val / test")
    print("="*55)
    X_val  = proc.transform(proc.val_idx)
    y_val  = proc.labels.iloc[proc.val_idx].reset_index(drop=True)

    X_test = proc.transform(proc.test_idx)
    y_test = proc.labels.iloc[proc.test_idx].reset_index(drop=True)

    print(f"  X_val  : {X_val.shape}   |  pos rate {y_val.mean():.3f}")
    print(f"  X_test : {X_test.shape}  |  pos rate {y_test.mean():.3f}")

    # ── Step 4: SMOTE on training set ────────────────────────────────────────
    # The processor already constructs 1:1 negatives, so SMOTE is optional.
    # Shown here for completeness; remove if neg_ratio already achieves balance.
    print("\n" + "="*55)
    print("  Step 4: SMOTE (optional, train only)")
    print("="*55)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {len(y_res):,} samples  (pos rate {y_res.mean():.3f})")

    # ── Step 5: Train baseline model ─────────────────────────────────────────
    print("\n" + "="*55)
    print("  Step 5: Train baseline models")
    print("="*55)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }

    results = []
    for name, model in models.items():
        print(f"\n  Training {name} …")
        model.fit(X_res, y_res)

        r_val  = evaluate(model, X_val,  y_val,  f"{name} — Val")
        r_test = evaluate(model, X_test, y_test, f"{name} — Test")
        results.extend([r_val, r_test])

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  Summary")
    print("="*55)
    summary = pd.DataFrame(results).set_index("tag").round(4)
    print(summary.to_string())

    return proc, X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    main()