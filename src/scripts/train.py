"""
src/scripts/train.py

Chicago Crime Prediction — PyTorch MLP Training Script
=======================================================

Pipeline
--------
    DataProcessor.load_and_split()
        → fit_transform_train()     fit + transform train split
        → transform(val_idx)        transform val  (no refit)
        → transform(test_idx)       transform test (no refit)
        → CrimeDataset → DataLoader (pin_memory for GPU speed)
        → MLP training with early stopping
        → Evaluation on val & test  (AUROC, AUPRC, classification report)

Design decisions
----------------
- No SMOTE: the DataProcessor already constructs a 1:1 balanced dataset via
  negative sampling.  Balanced batches via WeightedRandomSampler achieve the
  same effect per mini-batch without inflating dataset size.
- pos_weight in BCEWithLogitsLoss: secondary guard for any residual imbalance.
- pin_memory=True + non_blocking=True: overlap CPU→GPU transfers with compute.
- WeightedRandomSampler: ensures each mini-batch sees ~50% positives.

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
    roc_auc_score,
    average_precision_score,
    classification_report,
)
import warnings
warnings.filterwarnings("ignore")

from src.data.processor import DataProcessor

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

DATA_PATH    = PROJECT_ROOT / "data" / "raw" / "Crimes_-_2001_to_Present_20260216.csv"
RANDOM_STATE = 42

EPOCHS        = 50
BATCH_SIZE    = 2048
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 7
DROPOUT       = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class CrimeDataset(Dataset):
    """
    Converts preprocessed DataFrame + label Series to float32 tensors up-front.
    Per-sample indexing is then a pure array lookup — no casting overhead.
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X = torch.tensor(X.values, dtype=torch.float32)
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
    Four-layer MLP for binary crime prediction.

    Architecture
    ------------
    Input → FC(512) → BN → GELU → Dropout(0.3)
          → FC(256) → BN → GELU → Dropout(0.3)
          → FC(128) → BN → GELU → Dropout(0.3)
          → FC(64)  → BN → GELU
          → FC(1)                         [raw logit → BCEWithLogitsLoss]

    Skip connection: block2 output (256) projected to 64 and added to block4
    output, improving gradient flow through the deep stack.

    Output: raw logit.  Apply torch.sigmoid() at inference for probability.
    """

    def __init__(self, input_dim: int, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.block1   = self._fc_block(input_dim, 512, dropout)
        self.block2   = self._fc_block(512,       256, dropout)
        self.block3   = self._fc_block(256,       128, dropout)
        self.block4   = self._fc_block(128,        64, dropout=0.0)
        self.skip_proj = nn.Linear(256, 64, bias=False)
        self.head      = nn.Linear(64, 1)
        self._init_weights()

    @staticmethod
    def _fc_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        layers: list = [
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.block1(x)
        h2 = self.block2(x)
        x  = self.block3(h2)
        x  = self.block4(x) + self.skip_proj(h2)
        return self.head(x).squeeze(1)   # (B,)


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader builder
# ──────────────────────────────────────────────────────────────────────────────

def make_loader(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    shuffle: bool = False,
    use_weighted_sampler: bool = False,
) -> DataLoader:
    """
    Build a DataLoader.

    use_weighted_sampler=True (training only): WeightedRandomSampler ensures
    each mini-batch has ~50% positives regardless of overall class ratio.
    pin_memory=True overlaps CPU→GPU transfers with GPU compute.
    """
    dataset    = CrimeDataset(X, y)
    pin_memory = DEVICE.type == "cuda"

    if use_weighted_sampler:
        labels = y.values
        n_pos  = max(labels.sum(), 1)
        n_neg  = max(len(labels) - n_pos, 1)
        weights = torch.tensor(
            [1.0 / n_pos if l == 1 else 1.0 / n_neg for l in labels],
            dtype=torch.float32,
        )
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=4,
            pin_memory=pin_memory,
            persistent_workers=True,
        )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=pin_memory,
        persistent_workers=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:     CrimeMLP,
    loader:    DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler:    torch.cuda.amp.GradScaler,
) -> float:
    """Single training epoch. Returns mean batch loss."""
    model.train()
    total_loss, n_batches = 0.0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE.type,
                            enabled=(DEVICE.type == "cuda")):
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def predict_proba(model: CrimeMLP, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_prob, y_true) arrays from a DataLoader."""
    model.eval()
    probs, labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type,
                            enabled=(DEVICE.type == "cuda")):
            logits = model(X_batch)
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        labels.extend(y_batch.numpy())
    return np.array(probs), np.array(labels)


def evaluate_loader(model: CrimeMLP, loader: DataLoader, tag: str) -> dict:
    y_prob, y_true = predict_proba(model, loader)
    y_pred = (y_prob >= 0.5).astype(int)
    auroc  = roc_auc_score(y_true, y_prob)
    auprc  = average_precision_score(y_true, y_prob)

    print(f"\n{'='*55}\n  [{tag}]\n{'='*55}")
    print(f"  AUROC : {auroc:.4f}  |  AUPRC : {auprc:.4f}\n")
    print(classification_report(
        y_true, y_pred, target_names=["No Crime (0)", "Crime (1)"]
    ))
    return {"tag": tag, "auroc": auroc, "auprc": auprc}


def fit(
    model:        CrimeMLP,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    pos_weight:   float,
) -> list:
    """Training loop with cosine LR schedule and early stopping on val AUROC."""
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=EPOCHS, eta_min=1e-5
    )
    amp_scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_auroc, patience_count, best_state = -1.0, 0, None
    history = []

    print(f"\n{'='*55}")
    print(f"  Training on {DEVICE}  |  max {EPOCHS} epochs  |  patience {PATIENCE}")
    print(f"{'='*55}")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion,
                                 optimiser, amp_scaler)
        scheduler.step()

        val_probs, val_labels = predict_proba(model, val_loader)
        val_auroc = roc_auc_score(val_labels, val_probs)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_auroc": val_auroc})

        print(f"  Epoch {epoch:>3}/{EPOCHS}  "
              f"loss={train_loss:.4f}  val_auroc={val_auroc:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"patience={patience_count}/{PATIENCE}")

        if val_auroc > best_auroc:
            best_auroc, patience_count = val_auroc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n  ⏹  Early stop at epoch {epoch}  "
                      f"(best val AUROC = {best_auroc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  ✓  Best checkpoint restored  (val AUROC = {best_auroc:.4f})")

    return history


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    print(f"\n  Device : {DEVICE}")

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  Step 1: Data preprocessing")
    print("="*55)
    proc = DataProcessor(
        data_path=str(DATA_PATH),
        neg_ratio=1.0,
        random_state=RANDOM_STATE,
    )
    proc.load_and_split()

    X_train = proc.fit_transform_train()
    y_train = proc.labels.iloc[proc.train_idx].reset_index(drop=True)
    X_val   = proc.transform(proc.val_idx)
    y_val   = proc.labels.iloc[proc.val_idx].reset_index(drop=True)
    X_test  = proc.transform(proc.test_idx)
    y_test  = proc.labels.iloc[proc.test_idx].reset_index(drop=True)

    print(f"  X_train {X_train.shape}  pos={y_train.mean():.3f}")
    print(f"  X_val   {X_val.shape}    pos={y_val.mean():.3f}")
    print(f"  X_test  {X_test.shape}   pos={y_test.mean():.3f}")

    # ── Step 2: DataLoaders ───────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  Step 2: Building DataLoaders")
    print("="*55)
    train_loader = make_loader(X_train, y_train, use_weighted_sampler=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)
    print(f"  Train {len(train_loader)} batches  |  "
          f"Val {len(val_loader)}  |  Test {len(test_loader)}")

    # ── Step 3: Model ─────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  Step 3: Model")
    print("="*55)
    input_dim = X_train.shape[1]
    model     = CrimeMLP(input_dim=input_dim, dropout=DROPOUT).to(DEVICE)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Input dim  : {input_dim}")
    print(f"  Parameters : {n_params:,}")

    n_pos      = int(y_train.sum())
    n_neg      = len(y_train) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    print(f"  pos_weight : {pos_weight:.3f}  (n_pos={n_pos:,}  n_neg={n_neg:,})")

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    history = fit(model, train_loader, val_loader, pos_weight)

    # ── Step 5: Evaluate ──────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  Step 5: Final evaluation")
    print("="*55)
    val_result  = evaluate_loader(model, val_loader,  "Val  — CrimeMLP")
    test_result = evaluate_loader(model, test_loader, "Test — CrimeMLP")

    best_epoch = max(history, key=lambda r: r["val_auroc"])
    print("\n" + "="*55 + "\n  Summary\n" + "="*55)
    print(f"  Best epoch : {best_epoch['epoch']}  "
          f"val AUROC = {best_epoch['val_auroc']:.4f}")
    print(f"  Val   AUROC {val_result['auroc']:.4f}  "
          f"AUPRC {val_result['auprc']:.4f}")
    print(f"  Test  AUROC {test_result['auroc']:.4f}  "
          f"AUPRC {test_result['auprc']:.4f}")

    return model, proc, history


if __name__ == "__main__":
    main()