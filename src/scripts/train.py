"""
src/scripts/train.py

Chicago Crime Prediction — PyTorch MLP Training Script
=======================================================

Run
---
    uv run python src/scripts/train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
)
import warnings
warnings.filterwarnings("ignore")

from src.data.processor import DataProcessor

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH    = PROJECT_ROOT / "data" / "raw" / "Crimes_2023_to_2025.csv"
RANDOM_STATE = 42
EPOCHS       = 100          # more epochs affordable on small data
BATCH_SIZE   = 256          # ~50 batches per epoch on 13k train rows — good granularity
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 10           # more patience: small dataset has noisier val AUROC
DROPOUT      = 0.3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def _to_float32(X: pd.DataFrame) -> np.ndarray:
    """
    Convert DataFrame to float32 ndarray, coercing any residual non-numeric
    columns and reporting what was dropped.
    """
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        print(f"  ⚠  CrimeDataset: dropping object cols: {obj_cols}")
        X = X.drop(columns=obj_cols)
    arr = X.values.astype(np.float32)   # will raise if non-numeric types remain
    return arr


class CrimeDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X = torch.from_numpy(_to_float32(X))
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class CrimeMLP(nn.Module):
    """
    3-layer MLP sized for ~35k training rows (post-split, post-hist-drop).

    Architecture: Input → 128 → 64 → 32 → 1
    Smaller than the original 512→256→128→64 stack to avoid overfitting on
    50k total samples (~35k after split and cold-start row drops).

    Skip connection: block-1 output (128) projected to 32 and added to
    block-3 output before the head.
    Returns raw logit; apply sigmoid at inference.
    """
    def __init__(self, input_dim: int, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.b1        = self._block(input_dim, 128, dropout)
        self.b2        = self._block(128,         64, dropout)
        self.b3        = self._block(64,          32, 0.0)
        self.skip_proj = nn.Linear(128, 32, bias=False)
        self.head      = nn.Linear(32, 1)
        self._init()

    @staticmethod
    def _block(i: int, o: int, d: float) -> nn.Sequential:
        layers = [nn.Linear(i, o), nn.BatchNorm1d(o), nn.GELU()]
        if d > 0:
            layers.append(nn.Dropout(d))
        return nn.Sequential(*layers)

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.b1(x)
        x  = self.b2(h1)
        x  = self.b3(x) + self.skip_proj(h1)
        return self.head(x).squeeze(1)


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_loader(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    shuffle: bool = False,
    weighted: bool = False,
) -> DataLoader:
    """
    weighted=True  → WeightedRandomSampler (training split).
    """
    ds  = CrimeDataset(X, y)
    pin = DEVICE.type == "cuda"
    # num_workers=0: spawning worker processes has higher overhead than the
    # data loading itself at 50k rows; 0 = synchronous loading in main process.
    kw  = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=pin)

    if weighted:
        labels = y.values
        n_pos  = max(float(labels.sum()), 1)
        n_neg  = max(float(len(labels) - n_pos), 1)
        w = torch.tensor(
            [1.0 / n_pos if l == 1 else 1.0 / n_neg for l in labels],
            dtype=torch.float32,
        )
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        return DataLoader(ds, sampler=sampler, **kw)

    return DataLoader(ds, shuffle=shuffle, **kw)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def _train_epoch(model, loader, criterion, opt, scaler) -> float:
    model.train()
    total, n = 0.0, 0
    for Xb, yb in loader:
        Xb = Xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            loss = criterion(model(Xb), yb)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def _predict(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for Xb, yb in loader:
        Xb = Xb.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            logits = model(Xb)
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        labels.extend(yb.numpy())
    return np.array(probs), np.array(labels)


def _evaluate(model, loader, tag: str) -> dict:
    yp, yt = _predict(model, loader)
    auroc  = roc_auc_score(yt, yp)
    auprc  = average_precision_score(yt, yp)
    print(f"\n{'='*55}\n  [{tag}]\n{'='*55}")
    print(f"  AUROC : {auroc:.4f}  |  AUPRC : {auprc:.4f}\n")
    print(classification_report(yt, (yp >= 0.5).astype(int),
                                 target_names=["No Crime", "Crime"]))
    return {"tag": tag, "auroc": auroc, "auprc": auprc}


def fit(model, train_loader, val_loader, pos_weight: float) -> list:
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(DEVICE)
    )
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    amp  = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_auroc, patience, best_sd = -1.0, 0, None
    history = []

    print(f"\n{'='*55}")
    print(f"  Training on {DEVICE}  |  max {EPOCHS} epochs  |  patience {PATIENCE}")
    print("="*55)

    for ep in range(1, EPOCHS + 1):
        loss = _train_epoch(model, train_loader, criterion, opt, amp)
        sch.step()
        val_p, val_l = _predict(model, val_loader)
        va = roc_auc_score(val_l, val_p)
        history.append({"epoch": ep, "loss": loss, "val_auroc": va})
        print(f"  Ep {ep:>3}/{EPOCHS}  loss={loss:.4f}  val_auroc={va:.4f}  "
              f"lr={sch.get_last_lr()[0]:.1e}  pat={patience}/{PATIENCE}")
        if va > best_auroc:
            best_auroc, patience = va, 0
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"\n  ⏹  Early stop  (best val AUROC={best_auroc:.4f})")
                break

    if best_sd:
        model.load_state_dict(best_sd)
        print(f"  ✓  Best checkpoint restored  (val AUROC={best_auroc:.4f})")
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    print(f"\n  Device : {DEVICE}")

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    print("\n" + "="*55 + "\n  Step 1: Preprocessing\n" + "="*55)
    proc = DataProcessor(str(DATA_PATH), neg_ratio=1.0, random_state=RANDOM_STATE)
    proc.load_and_split()

    X_train, y_train = proc.fit_transform_train()
    X_val,   y_val   = proc.transform(proc.val_idx)
    X_test,  y_test  = proc.transform(proc.test_idx)

    print(f"  X_train {X_train.shape}  pos={y_train.mean():.3f}")
    print(f"  X_val   {X_val.shape}    pos={y_val.mean():.3f}")
    print(f"  X_test  {X_test.shape}   pos={y_test.mean():.3f}")

    # ── Step 2: DataLoaders ───────────────────────────────────────────────────
    print("\n" + "="*55 + "\n  Step 2: DataLoaders\n" + "="*55)
    train_loader = make_loader(X_train, y_train, weighted=True)
    val_loader   = make_loader(X_val,   y_val)
    test_loader  = make_loader(X_test,  y_test)
    print(f"  Batches  train={len(train_loader)}  "
          f"val={len(val_loader)}  test={len(test_loader)}")

    # ── Step 3: Model ─────────────────────────────────────────────────────────
    print("\n" + "="*55 + "\n  Step 3: Model\n" + "="*55)
    model    = CrimeMLP(input_dim=X_train.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_pos    = int(y_train.sum());  n_neg = len(y_train) - n_pos
    pw       = n_neg / max(n_pos, 1)
    print(f"  input_dim={X_train.shape[1]}  params={n_params:,}  pos_weight={pw:.3f}")

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    history = fit(model, train_loader, val_loader, pw)

    # ── Step 5: Evaluate ──────────────────────────────────────────────────────
    print("\n" + "="*55 + "\n  Step 5: Evaluation\n" + "="*55)
    vr = _evaluate(model, val_loader,  "Val  — CrimeMLP")
    tr = _evaluate(model, test_loader, "Test — CrimeMLP")

    best = max(history, key=lambda r: r["val_auroc"])
    print(f"\n  Best epoch : {best['epoch']}  val AUROC={best['val_auroc']:.4f}")
    print(f"  Val   AUROC={vr['auroc']:.4f}  AUPRC={vr['auprc']:.4f}")
    print(f"  Test  AUROC={tr['auroc']:.4f}  AUPRC={tr['auprc']:.4f}")
    return model, proc, history


if __name__ == "__main__":
    main()