#!/usr/bin/env python3
"""
pl_simple_baseline_train.py

This script wires together:
  - FlareDSDataset (downstream flare labeling on top of Surya/Helio stacks)
  - RegressionFlareModel (a minimal linear baseline over channel×time features)
  - FlareLightningModule (Lightning training/validation loop and logging)
  - Optional Weights & Biases (W&B) logging

It is designed to match the structure shown in `1_baseline_template.ipynb`, but in a
reproducible, command-line runnable form.

IMPORTANT: Make sure to set the visible device to the assigned GPU when you run this script.

Example:
  python script_0_baseline_template_train.py \
    --config ./configs/config.yaml \
    --ds_flare_index_path ./data/hek_flare_catalog.csv \
    --batch_size 2 \
    --max_epochs 2 \
    --cuda_visible_devices "0" \
    --use_wandb

"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

import yaml
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# W&B is optional; only imported when enabled.
try:
    from pytorch_lightning.loggers import WandbLogger  # type: ignore
except Exception:
    WandbLogger = None  # type: ignore


# Determine the absolute path to the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Determine the absolute path to the main project root
# (assuming script_0_baseline_template_train.py is in downstream_apps/your_downstream/)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Construct absolute paths to Surya and hfmds directories
SURYA_DIR = os.path.join(PROJECT_ROOT, "Surya")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SURYA_DIR not in sys.path:
    sys.path.insert(0, SURYA_DIR)


from surya.utils.data import build_scalers

from downstream_apps.template.datasets.template_dataset import FlareDSDataset
from downstream_apps.template.models.simple_baseline import RegressionFlareModel
from downstream_apps.template.metrics.template_metrics import FlareMetrics
from downstream_apps.template.lightning_modules.pl_simple_baseline import FlareLightningModule


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train the simple flare baseline with PyTorch Lightning.")

    # Core config
    p.add_argument("--config", type=str, default="./configs/config.yaml", help="Path to the YAML config file.")

    # Downstream flare label config (defaults match notebook)
    p.add_argument("--ds_flare_index_path", type=str, default="./data/hek_flare_catalog.csv",
                   help="Path to flare catalog CSV used for labels (e.g., HEK flare catalog).")
    p.add_argument("--ds_time_column", type=str, default="start_time",
                   help="Timestamp column in the flare catalog CSV (e.g., start_time).")
    p.add_argument("--ds_time_tolerance", type=str, default="4d",
                   help="Tolerance for merge_asof matching (e.g., '4d').")
    p.add_argument("--ds_match_direction", type=str, default="forward", choices=["forward", "backward", "nearest"],
                   help="Direction for merge_asof matching.")

    # Runtime / performance
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers. For W&B + notebooks, 0 is most robust.")
    p.add_argument("--pin_memory", action="store_true", help="Enable pinned-memory batches for faster GPU transfer.")
    p.add_argument("--mp_context", type=str, default="spawn", choices=["spawn", "fork", "forkserver"],
                   help="Multiprocessing context for DataLoader (Linux default is fork; spawn is safer with loggers).")

    # Trainer
    p.add_argument("--max_epochs", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=str, default="auto")
    p.add_argument("--log_every_n_steps", type=int, default=2)
    p.add_argument("--num_sanity_val_steps", type=int, default=2,
                   help="Lightning sanity check steps. Set 0 to disable.")

    # Output/logging
    p.add_argument("--run_dir", type=str, default="./runs/simple_flare", help="Root directory for logs/checkpoints.")
    p.add_argument("--csv_log_name", type=str, default="simple_flare", help="Name for CSVLogger subdir.")

    # W&B
    p.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb_project", type=str, default="template_flare_regression")
    p.add_argument("--wandb_entity", type=str, default="surya_handson")
    p.add_argument("--wandb_run_name", type=str, default="baseline_experiment_1")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"],
                   help="W&B mode. Use offline/disabled for restricted networks.")
    p.add_argument("--wandb_dir", type=str, default="./wandb/wandb_logs",
                   help="Local directory where W&B stores run files (must be writable).")

    # CUDA visibility convenience
    p.add_argument("--cuda_visible_devices", type=str, default="",
                   help="If set, exports CUDA_VISIBLE_DEVICES before torch initializes CUDA.")

    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config from disk."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_scalers_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build scalers using Surya's build_scalers, mirroring the notebook."""
    scalers_path = cfg["data"]["scalers_path"]
    with open(scalers_path, "r") as f:
        scaler_info = yaml.safe_load(f)
    return build_scalers(info=scaler_info)


def make_datasets(cfg: Dict[str, Any], scalers: Dict[str, Any], args: argparse.Namespace):
    """Construct train and validation datasets."""
    common_kwargs = dict(
        time_delta_input_minutes=cfg["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=cfg["data"]["time_delta_target_minutes"],
        n_input_timestamps=cfg["model"]["time_embedding"]["time_dim"],
        rollout_steps=cfg["rollout_steps"],
        channels=cfg["data"]["channels"],
        drop_hmi_probability=cfg["drop_hmi_probablity"],
        use_latitude_in_learned_flow=cfg["use_latitude_in_learned_flow"],
        scalers=scalers,
        s3_use_simplecache=True,
        s3_cache_dir="/tmp/helio_s3_cache",
        # Downstream-specific:
        return_surya_stack=True,
        max_number_of_samples=cfg.get("max_number_of_samples", 6),
        ds_flare_index_path=args.ds_flare_index_path,
        ds_time_column=args.ds_time_column,
        ds_time_tolerance=args.ds_time_tolerance,
        ds_match_direction=args.ds_match_direction,
    )

    train_dataset = FlareDSDataset(
        index_path=cfg["data"]["train_data_path"],
        phase="train",
        **common_kwargs,
    )

    val_dataset = FlareDSDataset(
        index_path=cfg["data"]["valid_data_path"],
        **common_kwargs,
    )

    return train_dataset, val_dataset


def make_dataloaders(train_dataset, val_dataset, args: argparse.Namespace):
    """Create DataLoaders for train and validation."""
    dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # Only pass multiprocessing_context/persistent_workers when workers are enabled.
    if args.num_workers > 0:
        dl_kwargs.update(
            multiprocessing_context=args.mp_context,
            persistent_workers=True,
        )

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        **dl_kwargs,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        **dl_kwargs,
    )

    return train_loader, val_loader


def make_model(cfg: Dict[str, Any], scalers: Dict[str, Any]) -> torch.nn.Module:
    """Instantiate the RegressionFlareModel baseline."""
    n_input_timestamps = cfg["model"]["time_embedding"]["time_dim"]
    n_channels = len(cfg["data"]["channels"])
    input_dim = n_input_timestamps * n_channels
    return RegressionFlareModel(input_dim, cfg["data"]["channels"], scalers)


def make_metrics() -> Dict[str, Any]:
    """Instantiate the FlareMetrics objects used by FlareLightningModule."""
    train_loss_metrics = FlareMetrics("train_loss")
    train_evaluation_metrics = FlareMetrics("train_metrics")
    validation_evaluation_metrics = FlareMetrics("val_metrics")

    return {
        "train_loss": train_loss_metrics,
        "train_metrics": train_evaluation_metrics,
        "val_metrics": validation_evaluation_metrics,
    }


def maybe_make_wandb_logger(args: argparse.Namespace):
    """Create a WandbLogger if enabled; otherwise return None."""
    if not args.use_wandb:
        return None

    if WandbLogger is None:
        raise ImportError("pytorch_lightning WandbLogger is not available; install wandb and lightning extras.")

    # Ensure writable directories for W&B artifacts/logs
    os.makedirs(args.wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = args.wandb_dir
    os.environ["WANDB_CACHE_DIR"] = os.path.join(args.wandb_dir, "cache")
    os.environ["WANDB_CONFIG_DIR"] = os.path.join(args.wandb_dir, "config")
    os.environ["TMPDIR"] = os.path.join(args.wandb_dir, "tmp")
    for d in [os.environ["WANDB_CACHE_DIR"], os.environ["WANDB_CONFIG_DIR"], os.environ["TMPDIR"]]:
        os.makedirs(d, exist_ok=True)

    # W&B mode control (online/offline/disabled)
    os.environ["WANDB_MODE"] = args.wandb_mode

    return WandbLogger(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        log_model=False,
        save_dir=args.wandb_dir,
    )


def main() -> None:
    """Entry point."""
    args = parse_args()

    if args.cuda_visible_devices is not None:
        # Must be set before CUDA is initialized for it to have effect.
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    pl.seed_everything(args.seed, workers=True)

    cfg = load_config(args.config)
    scalers = build_scalers_from_config(cfg)

    train_dataset, val_dataset = make_datasets(cfg, scalers, args)
    train_loader, val_loader = make_dataloaders(train_dataset, val_dataset, args)

    model = make_model(cfg, scalers)
    metrics = make_metrics()

    lit_model = FlareLightningModule(
        model=model,
        metrics=metrics,
        lr=args.learning_rate,
        batch_size=args.batch_size,
    )

    # Loggers
    csv_logger = CSVLogger(save_dir=args.run_dir, name=args.csv_log_name)
    wandb_logger = maybe_make_wandb_logger(args)

    logger_list = [csv_logger] + ([wandb_logger] if wandb_logger is not None else [])

    # Checkpointing
    ckpt = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices if args.devices == "auto" else int(args.devices),
        logger=logger_list,
        callbacks=[ckpt],
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=args.num_sanity_val_steps,
        default_root_dir=args.run_dir,
    )

    trainer.fit(lit_model, train_loader, val_loader)

    print(f"Best checkpoint: {ckpt.best_model_path}")


if __name__ == "__main__":
    main()
