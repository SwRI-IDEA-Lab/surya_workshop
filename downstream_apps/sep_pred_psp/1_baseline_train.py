#!/usr/bin/env python3
"""
Runnable linear-baseline trainer for the SEP-from-PSP downstream app.

This is the slim, script form of `1_baseline_template.ipynb`: it strips the
inline "does this shape look right / does this metric look right" checking
cells the notebook uses for teaching, and keeps only what's needed to fit the
model. Use the notebook when you want to inspect intermediate values; use this
when you just want to launch a run (repeatedly, with different overrides).

Run from the repo root, with the GPU selected via CUDA_VISIBLE_DEVICES:

    CUDA_VISIBLE_DEVICES=0 python -m downstream_apps.sep_pred_psp.1_baseline_train

All parameters live in config_script.yaml. The CLI overrides only what varies
between runs of the same config.
"""

from __future__ import annotations

import argparse
import os

# Must be set BEFORE torch is imported: cuBLAS reads this once, when it initializes, so
# setting it later has no effect. See config_script.yaml's training.deterministic comment.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from functools import partial
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from downstream_apps.sep_pred_psp.configs import load_sep_psp_ds_config
from downstream_apps.sep_pred_psp.datasets.label_transform import build_log_standardizer
from downstream_apps.sep_pred_psp.datasets.template_dataset import SepPspDSDataset
from downstream_apps.sep_pred_psp.lightning_modules.pl_simple_baseline import SepPspLightningModule
from downstream_apps.sep_pred_psp.lightning_modules.val_prediction_logger import (
    ValidationPredictionLogger,
)
from downstream_apps.sep_pred_psp.metrics.template_metrics import SepPspMetrics
from downstream_apps.sep_pred_psp.models.simple_baseline import (
    RegressionSepPspModel,
    destandardize_channels,
)
from workshop_infrastructure.assets import ensure_assets
from workshop_infrastructure.datasets.builders import build_helio_dataloaders
from workshop_infrastructure.utils import build_scalers

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "config_script.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Path to the run config YAML (default: this app's config_script.yaml).")
    parser.add_argument("--run-name", type=str, default=None,
                        help="WandB/CSV run name. Defaults to the config's job_id.")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging (CSV logging still happens).")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override training.max_epochs from the config YAML.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override training.batch_size from the config YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("medium")

    cfg = load_sep_psp_ds_config(args.config)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    L.seed_everything(cfg.seed, workers=True)

    # The linear baseline needs no backbone, so skip the 1.8 GB weights download.
    ensure_assets(cfg, which=["scalers"])
    scalers = build_scalers(info=cfg.data.scalers_path)

    # log10 + z-score, fitted on the training split only and shared by both splits.
    label_transform = build_log_standardizer(
        cfg.data.sep_psp_index_path, cfg.data.train_data_path, column=cfg.data.label_column
    )
    print(f"[label] {label_transform}")

    train_loader, val_loader = build_helio_dataloaders(
        cfg,
        SepPspDSDataset,
        scalers=scalers,
        return_surya_stack=True,
        # Sized per split: max_samples drives training, max_val_samples holds validation
        # fixed, so sweeping max_samples changes what the model learns from and not what
        # it is scored on.
        train_kwargs={"max_number_of_samples": cfg.data.max_samples},
        val_kwargs={"max_number_of_samples": cfg.data.val_samples},
        active_above=cfg.data.active_above,
        quiet_below=cfg.data.quiet_below,
        label_column=cfg.data.label_column,
        label_transform=label_transform,
        ds_sep_psp_index_path=cfg.data.sep_psp_index_path,
        ds_time_column=cfg.data.ds_time_column,
        ds_time_tolerance=cfg.data.ds_time_tolerance,
        ds_match_direction=cfg.data.ds_match_direction,
        ds_non_event_buffer=cfg.data.non_event_buffer,
        sample_seed=cfg.seed,
    )

    n_input_timestamps = cfg.model.time_embedding.time_dim
    n_channels = len(cfg.data.channels)
    model = RegressionSepPspModel(n_input_timestamps * n_channels)
    preprocess_fn = partial(destandardize_channels, channel_order=cfg.data.channels, scalers=scalers)

    metrics = {
        "train_loss": SepPspMetrics("train_loss"),
        # val_loss is what ModelCheckpoint monitors; val_metrics are reported only.
        "val_loss": SepPspMetrics("val_loss"),
        "train_metrics": SepPspMetrics("train_metrics"),
        "val_metrics": SepPspMetrics("val_metrics"),
    }
    lit_model = SepPspLightningModule(
        model, metrics, lr=cfg.learning_rate, batch_size=cfg.batch_size, preprocess_fn=preprocess_fn
    )

    run_name = args.run_name or cfg.job_id
    loggers = [CSVLogger("runs", name=cfg.wandb_project)]
    if not args.no_wandb:
        loggers.insert(0, WandbLogger(
            entity=cfg.wandb_entity,  # None = personal account; set in YAML for team runs
            project=cfg.wandb_project,
            name=run_name,
            log_model=False,
            save_dir="./wandb/wandb_tmp",
        ))

    Path(cfg.output.ckpt_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.output.ckpt_dir,
        filename=f"{run_name}-" + "{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # Per-epoch validation predictions to CSV + the best epoch's histograms to WandB.
    val_predictions_cb = ValidationPredictionLogger(
        output_dir=Path(cfg.output.ckpt_dir) / "val_predictions" / run_name,
        monitor="val_loss",
        mode="min",
        # The target is log10-z-scored: already log space and legitimately negative, so
        # the y axis stays linear and is named for what is actually plotted.
        label_name="log10 Jlinlin (z-scored)",
        log_y=False,
    )

    max_epochs = args.max_epochs if args.max_epochs is not None else cfg.max_epochs
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        logger=loggers,
        callbacks=[checkpoint_cb, val_predictions_cb],
        log_every_n_steps=2,
    )

    trainer.fit(lit_model, train_loader, val_loader)

    if checkpoint_cb.best_model_path:
        print(f"[CKPT] Best checkpoint: {checkpoint_cb.best_model_path}")
        if checkpoint_cb.best_model_score is not None:
            print(f"[CKPT] Best val_loss: {float(checkpoint_cb.best_model_score):.6f}")
    else:
        print("[CKPT] No best checkpoint was saved.")


if __name__ == "__main__":
    main()
