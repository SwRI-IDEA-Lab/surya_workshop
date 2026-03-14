#!/usr/bin/env python3
"""
Runnable finetuning script derived from `2_finetune_template_1D.ipynb`.

Design goals
- Config-driven: all hyperparameters live in config_script.yaml
- Minimal CLI: only --config is required; three optional overrides for dev convenience
- Multi-GPU capable (DDP) when run as a script

Assumptions
- You have already downloaded `scalers.yaml` + model weights (the notebook ran `download_scalers_and_weights.sh`).
- You run this script from the repo root and specify devices via CUDA_VISIBLE_DEVICES:
    CUDA_VISIBLE_DEVICES=0,1 python -m downstream_apps.template.3_finetune_template_1D \
        --config downstream_apps/template/configs/config_script.yaml

All other parameters (batch_size, max_epochs, S3 upload settings, etc.) are set in
the YAML. Pass --max-epochs N to override max_epochs for quick sweeps without editing
the file.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from downstream_apps.template.configs import TrainingConfig, load_config
from downstream_apps.template.datasets.template_dataset import FlareDSDataset
from downstream_apps.template.lightning_modules.pl_simple_baseline import FlareLightningModule
from downstream_apps.template.metrics.template_metrics import FlareMetrics
from workshop_infrastructure.utils import apply_peft_lora, build_scalers


# ---------------------------------------------------------------------------
# S3 upload callback
# ---------------------------------------------------------------------------

class UploadBestCheckpointToS3(L.Callback):
    """
    Uploads the best checkpoint to S3 whenever ModelCheckpoint records a new best.

    - No-op unless output.s3_bucket is set in the config.
    - Only runs from global rank 0 under DDP.
    - Uses a stable S3 key (output.s3_best_key) so the latest best is always
      at a predictable location.
    """

    def __init__(
        self,
        checkpoint_cb: ModelCheckpoint,
        bucket: str | None,
        prefix: str = "",
        fixed_key_name: str | None = "best.ckpt",
    ):
        super().__init__()
        self.checkpoint_cb = checkpoint_cb
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.fixed_key_name = fixed_key_name
        self.last_uploaded_best = None
        self._s3 = None

    def _upload_if_new_best(self, trainer) -> None:
        if not self.bucket:
            return
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        best_path = getattr(self.checkpoint_cb, "best_model_path", None)
        if not best_path or best_path == self.last_uploaded_best:
            return

        ckpt_path = Path(best_path)
        if not ckpt_path.exists():
            return

        if self._s3 is None:
            try:
                import boto3
            except ImportError as e:
                raise RuntimeError(
                    "boto3 is required for S3 uploads. Install with: pip install boto3"
                ) from e
            self._s3 = boto3.client("s3")

        object_name = self.fixed_key_name or ckpt_path.name
        s3_key = f"{self.prefix}/{object_name}" if self.prefix else object_name

        print(f"[S3] Uploading {ckpt_path} -> s3://{self.bucket}/{s3_key}")
        self._s3.upload_file(str(ckpt_path), self.bucket, s3_key)
        print("[S3] Upload complete.")
        self.last_uploaded_best = str(ckpt_path)

    def on_validation_end(self, trainer, pl_module) -> None:
        self._upload_if_new_best(trainer)

    def on_fit_end(self, trainer, pl_module) -> None:
        # Final check in case the last best update happens near fit end.
        self._upload_if_new_best(trainer)


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config.yaml",
                        help="Path to config_script.yaml.")
    # Dev toggles: flip without editing the YAML
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging (useful for local runs).")
    parser.add_argument("--train_baseline", action="store_true",
                        help="Train the simple linear baseline instead of HelioSpectformer.")
    # Sweep override: vary across jobs without touching the YAML
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override max_epochs from the config YAML.")
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


def build_datasets(cfg: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from config."""
    scalers = build_scalers(info=cfg.data.scalers_path)

    common_ds_kwargs = dict(
        time_delta_input_minutes=cfg.data.time_delta_input_minutes,
        time_delta_target_minutes=cfg.data.time_delta_target_minutes,
        n_input_timestamps=cfg.model.time_embedding.time_dim,
        rollout_steps=cfg.rollout_steps,
        channels=cfg.data.channels,
        drop_hmi_probability=cfg.drop_hmi_probability,
        use_latitude_in_learned_flow=cfg.use_latitude_in_learned_flow,
        scalers=scalers,
        s3_use_simplecache=False,
        s3_download_to_temp=True,
        s3_storage_options={"anon": cfg.data.s3_anon},
        s3_cache_dir=cfg.data.s3_cache_dir,
        # Downstream-specific
        return_surya_stack=True,
        max_number_of_samples=cfg.data.max_samples,
        label_transform=_flare_label_transform,
        ds_flare_index_path=cfg.data.flare_index_path,
        ds_time_column=cfg.data.ds_time_column,
        ds_time_tolerance=cfg.data.ds_time_tolerance,
        ds_match_direction=cfg.data.ds_match_direction,
    )

    train_dataset = FlareDSDataset(index_path=cfg.data.train_data_path, phase="train", **common_ds_kwargs)
    val_dataset = FlareDSDataset(index_path=cfg.data.valid_data_path, phase="val", **common_ds_kwargs)

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=8,
        multiprocessing_context="spawn",
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader


def build_model(cfg: TrainingConfig, train_baseline: bool = False) -> L.LightningModule:
    """Instantiate the model and wrap it in a LightningModule."""
    metrics = {
        "train_loss": FlareMetrics("train_loss"),
        "train_metrics": FlareMetrics("train_metrics"),
        "val_metrics": FlareMetrics("val_metrics"),
    }

    if train_baseline:
        from functools import partial
        from downstream_apps.template.models.simple_baseline import (
            RegressionFlareModel,
            inverse_transform_channels,
        )
        scalers = build_scalers(info=cfg.data.scalers_path)
        n_input_timestamps = cfg.model.time_embedding.time_dim
        n_channels = len(cfg.data.channels)
        model = RegressionFlareModel(n_input_timestamps * n_channels)
        preprocess_fn = partial(inverse_transform_channels, channel_order=cfg.data.channels, scalers=scalers)
        return FlareLightningModule(model, metrics, lr=cfg.learning_rate, batch_size=cfg.batch_size, preprocess_fn=preprocess_fn)
    else:
        from workshop_infrastructure.models.finetune_models import HelioSpectformer1D
        model = HelioSpectformer1D.from_config(
            cfg.model,
            num_outputs=1,
            dtype=cfg.dtype,
            use_latitude_in_learned_flow=cfg.use_latitude_in_learned_flow,
        )
        _load_pretrained_weights(model, cfg.pretrained_path)
        if cfg.model.use_lora:
            model = apply_peft_lora(model, cfg.model.lora_config)

    return FlareLightningModule(model, metrics, lr=cfg.learning_rate, batch_size=cfg.batch_size)


def _load_pretrained_weights(model: torch.nn.Module, pretrained_path: str | None) -> None:
    """Load pretrained weights into model, skipping shape-mismatched keys.

    The pretrained checkpoint was saved from HelioSpectFormer directly, so its
    keys are flat (e.g. ``embedding.proj.weight``).  The composed fine-tuning
    models nest that backbone under ``backbone.*``, so we try both the original
    key and the ``backbone.``-prefixed key when matching against the current
    model's state dict.
    """
    if not pretrained_path:
        return
    print(f"Loading pretrained weights from {pretrained_path}.")
    model_state = model.state_dict()
    checkpoint_state = torch.load(pretrained_path, weights_only=True, map_location="cpu")

    # Build a remapped view that also tries the backbone. prefix.
    remapped = {}
    for k, v in checkpoint_state.items():
        for candidate in (k, f"backbone.{k}"):
            if candidate in model_state and hasattr(v, "shape") and v.shape == model_state[candidate].shape:
                remapped[candidate] = v
                break

    model_state.update(remapped)
    model.load_state_dict(model_state, strict=True)


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
    L.seed_everything(42, workers=True)

    cfg = load_config(args.config)
    train_loader, val_loader = build_datasets(cfg)
    lit_model = build_model(cfg, train_baseline=args.train_baseline)
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
