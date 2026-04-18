#!/usr/bin/env python3
"""Run the integrated Jiadong STGNN experiments."""
from __future__ import annotations

import argparse

from src.experimental.jiadong_stgnn.config import (
    CHECKPOINT_DIR,
    GNN_HIDDEN_DIM,
    TEMPORAL_HIDDEN_DIM,
    WANDB_PROJECT,
)
from src.experimental.jiadong_stgnn.evaluate import evaluate, print_metrics
from src.experimental.jiadong_stgnn.models.stgnn import STGNN
from src.experimental.jiadong_stgnn.train import train_model
from src.experimental.jiadong_stgnn.utils.experiment import prepare_data, save_run_results

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def run_single_stgnn(
    temporal_type: str,
    train_loader,
    val_loader,
    test_loader,
    adj_tensor,
    scaler,
    use_wandb: bool = False,
):
    """Run one STGNN experiment variant."""
    run_name = f"STGNN-{temporal_type.upper()}"
    print(f"\n{'=' * 60}")
    print(f"  {run_name}")
    print(f"{'=' * 60}")

    if use_wandb and HAS_WANDB:
        wandb.init(
            project=WANDB_PROJECT.split("/")[-1],
            entity=WANDB_PROJECT.split("/")[0],
            name=run_name,
            config={
                "model_type": "stgnn",
                "temporal": temporal_type,
                "gnn_hidden": GNN_HIDDEN_DIM,
                "temporal_hidden": TEMPORAL_HIDDEN_DIM,
            },
            reinit=True,
        )

    model = STGNN(temporal_type=temporal_type)
    n_params = sum(param.numel() for param in model.parameters())
    print(f"Parameters: {n_params:,}")

    history = train_model(
        model,
        train_loader,
        val_loader,
        adj_tensor,
        save_path=CHECKPOINT_DIR / f"{run_name}.pt",
        use_wandb=use_wandb,
    )

    val_metrics = evaluate(model, val_loader, adj_tensor, scaler)
    print_metrics(val_metrics, f"{run_name} Validation")

    test_metrics = evaluate(model, test_loader, adj_tensor, scaler)
    print_metrics(test_metrics, f"{run_name} Test")

    if use_wandb and HAS_WANDB:
        wandb.log({f"test/{key}": value for key, value in test_metrics.items() if isinstance(value, (int, float))})
        wandb.finish()

    save_run_results(run_name, history, val_metrics, test_metrics)
    return run_name, history, val_metrics, test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Jiadong's integrated STGNN experiments")
    parser.add_argument(
        "--temporal",
        type=str,
        default="all",
        choices=["lstm", "gru", "mha", "all"],
        help="Temporal encoder type (default: all)",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild cached scaler and adjacency assets from scratch",
    )
    args = parser.parse_args()

    train_loader, val_loader, test_loader, adj_tensor, scaler = prepare_data(
        force_rebuild=args.force_rebuild
    )
    temporal_types = ["lstm", "gru", "mha"] if args.temporal == "all" else [args.temporal]

    for temporal_type in temporal_types:
        run_single_stgnn(
            temporal_type,
            train_loader,
            val_loader,
            test_loader,
            adj_tensor,
            scaler,
            use_wandb=args.wandb,
        )

    print("\nDone: all requested Jiadong STGNN experiments completed.")


if __name__ == "__main__":
    main()

