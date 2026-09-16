#!/usr/bin/env python3
"""
Runnable finetuning script derived from `2_finetune_template_1D.ipynb`.

Design goals
- Config-driven: all hyperparameters live in config_script.yaml
- Minimal CLI: --config plus a handful of per-run overrides
- Multi-GPU capable (DDP) when run as a script

Assumptions
- Assets (`scalers.yaml` + model weights) are downloaded automatically on first run.
- You run this script from the repo root and specify devices via CUDA_VISIBLE_DEVICES:
    CUDA_VISIBLE_DEVICES=0,1 python -m downstream_apps.template.3_finetune_template_1D \
        --config downstream_apps/template/configs/config_script.yaml

All parameters live in the YAML. The CLI overrides only what genuinely varies between
runs of the same config: --max-epochs and --batch-size (sweeps), --s3-cache-dir
(per-machine scratch) and --deterministic (reproducibility, off by default for speed).
Everything else is a config edit.

Forking this script: build_datasets() and build_model() are the only two functions with
task-specific content. build_trainer() and main() should need no changes.
"""

from __future__ import annotations

import argparse
import os

# Must be set BEFORE torch is imported: cuBLAS reads this once, when it initializes, so
# setting it later has no effect. Deterministic cuBLAS on CUDA >= 10.2 requires it, and
# without it every run under training.deterministic warns (or raises, when set to true).
# setdefault so a deliberate ":16:8" (smaller workspace, slightly slower) is respected.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
from typing import Tuple

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from downstream_apps.template.configs import TrainingConfig, load_flare_config
from downstream_apps.template.datasets.template_dataset import FlareDSDataset
from downstream_apps.template.lightning_modules.pl_simple_baseline import FlareLightningModule
from downstream_apps.template.metrics.template_metrics import FlareMetrics
from workshop_infrastructure.assets import ensure_assets
from workshop_infrastructure.datasets.builders import build_helio_dataloaders
from workshop_infrastructure.utils import (
    apply_peft_lora,
    build_scalers,
    load_pretrained_weights,
    UploadBestCheckpointToS3,
)

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "config_script.yaml"

# --deterministic accepts the same three tokens as the YAML key. argparse hands back a
# string, so the two boolean ones are mapped to real bools -- TrainingConfig validates
# against True/False/"warn", not against their spellings.
_DETERMINISTIC_CLI = {"false": False, "warn": "warn", "true": True}


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Path to the run config YAML (default: this app's config_script.yaml).")
    # Dev toggles: flip without editing the YAML
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging (useful for local runs).")
    parser.add_argument("--train_baseline", action="store_true",
                        help="Train the simple linear baseline instead of HelioSpectformer.")
    # Per-job / per-machine overrides: vary across runs without touching the YAML
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override training.max_epochs from the config YAML.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override training.batch_size from the config YAML.")
    parser.add_argument("--s3-cache-dir", type=str, default=None,
                        help="Override data.s3_cache_dir (the local cache for S3 reads). "
                             "Handy when the same config runs on machines with different scratch.")
    parser.add_argument("--deterministic", choices=tuple(_DETERMINISTIC_CLI), default=None,
                        help="Override training.deterministic. The config default is 'false', "
                             "which trades reproducibility for roughly 20%% throughput. Pass "
                             "'warn' when you need to tell whether a change in your results came "
                             "from your edit or from run-to-run drift.")
    return parser.parse_args()


def _flare_label_transform(intensity: "pd.Series") -> "pd.Series":
    """Normalize flare peak intensity for the template task.

    Converts raw GOES intensity to a z-score-like label:
      1. Take log10 (intensity values span many orders of magnitude).
      2. Shift so the minimum is 0.
      3. Scale by 2 * std so most values fall in [-1, 1].
    """
    import numpy as np
    log_intensity = np.log10(intensity)
    shifted = log_intensity - log_intensity.min()
    return shifted / (2 * shifted.std())


def build_datasets(cfg: TrainingConfig, scalers) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from config.

    Everything generic (channels, temporal sampling, S3 access, worker settings) is
    handled by build_helio_dataloaders(). Only the flare-specific arguments below are
    this app's business — when you fork the template, this is the list you replace.

    ``scalers`` is built once in main() and shared with build_model(), so the two paths
    cannot end up with different normalization statistics.
    """
    return build_helio_dataloaders(
        cfg,
        FlareDSDataset,
        scalers=scalers,
        seed=cfg.seed,
        return_surya_stack=True,
        max_number_of_samples=cfg.data.max_samples,
        label_transform=_flare_label_transform,
        ds_flare_index_path=cfg.data.flare_index_path,
        ds_time_column=cfg.data.ds_time_column,
        ds_time_tolerance=cfg.data.ds_time_tolerance,
        ds_match_direction=cfg.data.ds_match_direction,
    )


def build_model(cfg: TrainingConfig, scalers, train_baseline: bool = False) -> L.LightningModule:
    """Instantiate the model and wrap it in a LightningModule.

    ``scalers`` is only needed by the linear baseline, which consumes its inputs in
    signum-log space; the HelioSpectformer path works directly on normalized inputs.
    """
    metrics = {
        "train_loss": FlareMetrics("train_loss"),
        # val_loss is what ModelCheckpoint monitors; val_metrics are reported only.
        "val_loss": FlareMetrics("val_loss"),
        "train_metrics": FlareMetrics("train_metrics"),
        "val_metrics": FlareMetrics("val_metrics"),
    }

    if train_baseline:
        from functools import partial
        from downstream_apps.template.models.simple_baseline import (
            RegressionFlareModel,
            destandardize_channels,
        )
        n_input_timestamps = cfg.model.time_embedding.time_dim
        n_channels = len(cfg.data.channels)
        model = RegressionFlareModel(n_input_timestamps * n_channels)
        preprocess_fn = partial(destandardize_channels, channel_order=cfg.data.channels, scalers=scalers)
        return FlareLightningModule(model, metrics, lr=cfg.learning_rate, batch_size=cfg.batch_size, preprocess_fn=preprocess_fn)
    else:
        from workshop_infrastructure.models.finetune_models import HelioSpectformer1D
        model = HelioSpectformer1D.from_config(
            cfg.model,
            num_outputs=1,
            dtype=cfg.dtype,
            use_latitude_in_learned_flow=cfg.use_latitude_in_learned_flow,
        )
        load_pretrained_weights(model, cfg.model.pretrained_path)

        # Three fine-tuning regimes, selected from the model: section of the YAML:
        #   use_lora: true                        -> LoRA adapters (default)
        #   use_lora: false, freeze_backbone: true  -> linear probe (head only)
        #   use_lora: false, freeze_backbone: false -> full fine-tuning
        if cfg.model.freeze_backbone:
            for name, param in model.named_parameters():
                if name.startswith("backbone."):
                    param.requires_grad = False
        if cfg.model.use_lora:
            model = apply_peft_lora(model, cfg.model.lora_config)

        _log_trainable_parameters(model)

    return FlareLightningModule(model, metrics, lr=cfg.learning_rate, batch_size=cfg.batch_size)


def _log_trainable_parameters(model) -> None:
    """Print the trainable/total parameter counts, so the chosen regime is visible in the log."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    print(f"[MODEL] Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")


def build_trainer(
    cfg: TrainingConfig,
    no_wandb: bool = False,
    max_epochs_override: int | None = None,
) -> Tuple[L.Trainer, ModelCheckpoint]:
    """Configure loggers, callbacks, and the Lightning Trainer."""
    max_epochs = max_epochs_override if max_epochs_override is not None else cfg.max_epochs

    loggers = []
    if not no_wandb:
        loggers.append(WandbLogger(
            entity=cfg.wandb_entity,  # None = personal account; set in YAML for team runs
            project=cfg.wandb_project,
            name=cfg.job_id,
            log_model=False,
            save_dir=os.environ.get("TMPDIR", "./wandb/wandb_tmp"),
        ))
    loggers.append(CSVLogger("runs", name=cfg.job_id))

    Path(cfg.output.ckpt_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.output.ckpt_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    upload_cb = UploadBestCheckpointToS3(
        checkpoint_cb=checkpoint_cb,
        bucket=cfg.output.s3_bucket,
        prefix=cfg.output.s3_prefix,
        fixed_key_name=(cfg.output.s3_best_key or None),
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        # Reproducibility. "warn" (the default) gives bit-identical runs wherever a
        # deterministic kernel exists and names the op where one does not, instead of
        # killing the run. benchmark is pinned rather than inherited: cuDNN autotuning
        # picks algorithms by timing, so leaving it on would reintroduce run-to-run drift.
        deterministic=cfg.deterministic,
        benchmark=False,
        logger=loggers,
        callbacks=[checkpoint_cb, upload_cb],
        log_every_n_steps=2,
    )
    return trainer, checkpoint_cb


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("medium")

    cfg = load_flare_config(args.config)
    # Seeding comes after the config load, so the seed is a configured value rather than
    # a constant buried in the code. Seeds Python, NumPy and torch in this process;
    # workers=True extends it to DataLoader workers.
    L.seed_everything(cfg.seed, workers=True)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.s3_cache_dir is not None:
        cfg.data.s3_cache_dir = args.s3_cache_dir
    if args.deterministic is not None:
        cfg.deterministic = _DETERMINISTIC_CLI[args.deterministic]
    # Fetch scalers, and the backbone weights unless we are training the baseline.
    ensure_assets(cfg, which=["scalers"] if args.train_baseline else ["scalers", "weights"])

    # Built once and shared: the dataset normalizes with these, and the linear baseline
    # de-standardizes with them. Two separate builds could silently disagree.
    scalers = build_scalers(info=cfg.data.scalers_path)

    train_loader, val_loader = build_datasets(cfg, scalers)
    lit_model = build_model(cfg, scalers, train_baseline=args.train_baseline)
    trainer, checkpoint_cb = build_trainer(cfg, no_wandb=args.no_wandb, max_epochs_override=args.max_epochs)

    trainer.fit(lit_model, train_loader, val_loader)

    if checkpoint_cb.best_model_path:
        print(f"[CKPT] Best checkpoint: {checkpoint_cb.best_model_path}")
        if checkpoint_cb.best_model_score is not None:
            print(f"[CKPT] Best val_loss: {float(checkpoint_cb.best_model_score):.6f}")
    else:
        print("[CKPT] No best checkpoint was saved.")


if __name__ == "__main__":
    main()
