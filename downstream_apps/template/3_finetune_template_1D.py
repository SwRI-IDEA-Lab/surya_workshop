#!/usr/bin/env python3
"""
Runnable finetuning script derived from `2_finetune_template_1D.ipynb`.

Design goals
- Minimal CLI arguments
- Multi-GPU capable (DDP) when run as a script
- Mirrors notebook behavior closely

Assumptions
- You have already downloaded `scalers.yaml` + model weights (the notebook ran `download_scalers_and_weights.sh`).
- You run this script from the repo root and specify devices via CUDA_VISIBLE_DEVICES:
    CUDA_VISIBLE_DEVICES=0,1 python -m downstream_apps.template.3_finetune_template_1D \
        --config downstream_apps/template/configs/config_script.yaml \
        --batch-size 2 --max-epochs 20

S3 upload example
    CUDA_VISIBLE_DEVICES=0,1 python -m downstream_apps.template.3_finetune_template_1D \
        --config downstream_apps/template/configs/config_script.yaml \
        --batch-size 2 --max-epochs 20 \
        --s3_bucket my-ml-artifacts \
        --s3_prefix flare/exp_001 \
        --s3_best_key best.ckpt
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

    - No-op unless --s3_bucket is provided.
    - Only runs from global rank 0 under DDP.
    - Uses a stable S3 key by default (e.g. best.ckpt) so the latest best is
      always at a predictable location.
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
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size under DDP.")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--train_baseline", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory for local file cache.")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Directory to save local checkpoints.")
    parser.add_argument("--s3_bucket", type=str, default=None, help="Optional S3 bucket for best-checkpoint upload.")
    parser.add_argument("--s3_prefix", type=str, default="", help="Optional S3 key prefix (folder path).")
    parser.add_argument(
        "--s3_best_key",
        type=str,
        default="best.ckpt",
        help='Stable S3 object name for latest best checkpoint (set "" to use the local filename).',
    )
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


def build_datasets(
    cfg: TrainingConfig, args: argparse.Namespace
) -> Tuple[DataLoader, DataLoader]:
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
        # Downstream-specific
        return_surya_stack=True,
        max_number_of_samples=10,
        label_transform=_flare_label_transform,
        ds_flare_index_path=cfg.data.flare_index_path,
        ds_time_column="start_time",
        ds_time_tolerance="4d",
        ds_match_direction="forward",
        **({"s3_cache_dir": args.cache_dir} if args.cache_dir is not None else {}),
    )

    train_dataset = FlareDSDataset(index_path=cfg.data.train_data_path, phase="train", **common_ds_kwargs)
    val_dataset = FlareDSDataset(index_path=cfg.data.valid_data_path, phase="val", **common_ds_kwargs)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=8,
        multiprocessing_context="spawn",
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader


def build_model(cfg: TrainingConfig, args: argparse.Namespace) -> L.LightningModule:
    """Instantiate the model and wrap it in a LightningModule."""
    metrics = {
        "train_loss": FlareMetrics("train_loss"),
        "train_metrics": FlareMetrics("train_metrics"),
        "val_metrics": FlareMetrics("val_metrics"),
    }

    if args.train_baseline:
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
        return FlareLightningModule(model, metrics, lr=cfg.learning_rate, batch_size=args.batch_size, preprocess_fn=preprocess_fn)
    else:
        from workshop_infrastructure.models.finetune_models import HelioSpectformer1D
        m = cfg.model
        model = HelioSpectformer1D(
            img_size=m.img_size,
            patch_size=m.patch_size,
            in_chans=m.in_channels,
            embed_dim=m.embed_dim,
            time_embedding=vars(m.time_embedding),
            depth=m.depth,
            num_heads=m.num_heads,
            mlp_ratio=m.mlp_ratio,
            drop_rate=m.drop_rate,
            dtype=cfg.dtype,
            window_size=m.window_size,
            dp_rank=m.dp_rank,
            learned_flow=m.learned_flow,
            use_latitude_in_learned_flow=cfg.use_latitude_in_learned_flow,
            init_weights=m.init_weights,
            checkpoint_layers=m.checkpoint_layers,
            n_spectral_blocks=m.spectral_blocks,
            rpe=m.rpe,
            ensemble=m.ensemble,
            nglo=m.nglo,
            dropout=m.dropout,
            num_outputs=1,
            pooling=m.pooling,
            penultimate_linear_layer=m.penultimate_linear_layer,
        )
        _load_pretrained_weights(model, cfg.pretrained_path)
        if m.use_lora:
            model = apply_peft_lora(model, m.lora_config)

    return FlareLightningModule(model, metrics, lr=cfg.learning_rate, batch_size=args.batch_size)


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


def build_trainer(cfg: TrainingConfig, args: argparse.Namespace) -> Tuple[L.Trainer, ModelCheckpoint]:
    """Configure loggers, callbacks, and the Lightning Trainer."""
    loggers = []
    if not args.no_wandb:
        loggers.append(WandbLogger(
            entity="surya_handson",
            project=cfg.wandb_project,
            name=cfg.job_id,
            log_model=False,
            save_dir=os.environ.get("TMPDIR", "./wandb/wandb_tmp"),
        ))
    loggers.append(CSVLogger("runs", name=cfg.job_id))

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    upload_cb = UploadBestCheckpointToS3(
        checkpoint_cb=checkpoint_cb,
        bucket=args.s3_bucket,
        prefix=args.s3_prefix,
        fixed_key_name=(args.s3_best_key or None),
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
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
    train_loader, val_loader = build_datasets(cfg, args)
    lit_model = build_model(cfg, args)
    trainer, checkpoint_cb = build_trainer(cfg, args)

    trainer.fit(lit_model, train_loader, val_loader)

    if checkpoint_cb.best_model_path:
        print(f"[CKPT] Best checkpoint: {checkpoint_cb.best_model_path}")
        if checkpoint_cb.best_model_score is not None:
            print(f"[CKPT] Best val_loss: {float(checkpoint_cb.best_model_score):.6f}")
    else:
        print("[CKPT] No best checkpoint was saved.")


if __name__ == "__main__":
    main()
