#!/usr/bin/env python3
"""
Fine-tune Surya to predict an HMI magnetogram from three AIA EUV channels.

Derived from `2_euv2mag_finetune.ipynb`. Replaces the original
`3_euv2mag_100train_20val_pilot.py`, whose 100-train/20-val regime now lives in the YAML
as data.max_train_samples / data.max_val_samples rather than in the filename.

Design goals
- Config-driven: all hyperparameters live in configs/config_script.yaml
- Minimal CLI: --config plus a handful of per-run overrides
- Multi-GPU capable (DDP) when run as a script

Assumptions
- Assets (`scalers.yaml` + model weights) are downloaded automatically on first run.
- You run this from the repo root and select devices with CUDA_VISIBLE_DEVICES:
    CUDA_VISIBLE_DEVICES=0,1 python -m downstream_apps.Imagetranslation.3_finetune_euv2mag

What is task-specific here is build_datasets(), build_model() and the evaluation figure at
the end. build_trainer() and the body of main() are the template's, unchanged.

The input side is the unusual part of this app: Surya was pretrained on 13 channels and
this feeds it 3, so the tokenizer is built for 3 and initialized from the corresponding
slice of the pretrained convolution. See models/finetune_model.py for why that slice must
then be trainable, and MIGRATION.md for what it replaced.
"""

from __future__ import annotations

import argparse
import os

# Must be set BEFORE torch is imported: cuBLAS reads this once, when it initializes, so
# setting it later has no effect. Deterministic cuBLAS on CUDA >= 10.2 requires it, and
# without it every run under training.deterministic warns (or raises, when set to true).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
from typing import Tuple

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from downstream_apps.Imagetranslation.configs import TrainingConfig, load_euv2mag_config
from downstream_apps.Imagetranslation.datasets.euv2mag_dataset import Euv2MagDataset
from downstream_apps.Imagetranslation.lightning_modules.pl_euv2mag import (
    Euv2MagFinetuneLightningModule,
    Euv2MagLightningModule,
)
from downstream_apps.Imagetranslation.metrics.euv2mag_metrics import Euv2MagMetrics
from workshop_infrastructure.assets import ensure_assets
from workshop_infrastructure.datasets.builders import build_helio_dataloaders
from workshop_infrastructure.utils import (
    UploadBestCheckpointToS3,
    apply_peft_lora,
    build_scalers,
    load_pretrained_weights,
    resolve_trainable_backbone_modules,
)

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "config_script.yaml"

# argparse cannot express "false | warn | true" as mixed types, so the flag is a string
# and this maps it back to what Lightning's Trainer expects.
_DETERMINISTIC_CLI = {"false": False, "warn": "warn", "true": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Path to the YAML config (defaults to this app's own).")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override training.max_epochs from the config YAML.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override training.batch_size from the config YAML.")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Override data.max_train_samples. This is the knob for a "
                             "data-scaling sweep: the validation set is capped separately "
                             "by data.max_val_samples and stays fixed across runs.")
    parser.add_argument("--s3-cache-dir", type=str, default=None,
                        help="Override data.s3_cache_dir (the local cache for S3 reads).")
    parser.add_argument("--deterministic", choices=sorted(_DETERMINISTIC_CLI), default=None,
                        help="Override training.deterministic for this run.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging.")
    parser.add_argument("--train_baseline", action="store_true",
                        help="Train the 1x1-conv baseline instead of fine-tuning Surya.")
    parser.add_argument("--no-figure", action="store_true",
                        help="Skip the truth/prediction/residual figure after training.")
    return parser.parse_args()


def build_datasets(cfg: TrainingConfig, scalers) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from config.

    ``build_helio_dataloaders`` maps the config onto the ~20 ``HelioNetCDFDataset``
    arguments, so only this app's own kwargs appear here. The per-split size caps, the
    resolution (``data.pooling``) and the seeding are applied by the builder from the
    config, which is why they are absent from this list.
    """
    return build_helio_dataloaders(
        cfg,
        Euv2MagDataset,
        scalers=scalers,
        seed=cfg.seed,
        input_channels=cfg.data.input_channels,
        target_channels=cfg.data.target_channels,
    )


def build_model(cfg: TrainingConfig, train_baseline: bool = False) -> L.LightningModule:
    """Build the Lightning module: either the 1x1-conv baseline or the Surya fine-tune."""
    metrics = {
        mode: Euv2MagMetrics(mode)
        for mode in ("train_loss", "val_loss", "train_metrics", "val_metrics")
    }

    if train_baseline:
        from downstream_apps.Imagetranslation.models.simple_baseline import (
            Conv2DImageTranslationModel,
        )
        model = Conv2DImageTranslationModel(
            input_channels=cfg.data.input_channels,
            target_channels=cfg.data.target_channels,
            n_input_timestamps=cfg.model.time_embedding.time_dim,
        )
        # Plain Adam: there is no pretrained backbone here, so there is nothing to give a
        # different learning rate to.
        return Euv2MagLightningModule(
            model, metrics, lr=cfg.learning_rate, batch_size=cfg.batch_size
        )

    # The app's own model -- models/finetune_model.py. Edit the head there; you do not need
    # to touch workshop_infrastructure/ to change the architecture.
    from downstream_apps.Imagetranslation.models.finetune_model import Euv2MagSuryaModel
    model = Euv2MagSuryaModel(
        cfg.model,
        out_chans=len(cfg.data.target_channels),
        use_latitude_in_learned_flow=cfg.use_latitude_in_learned_flow,
    )

    # Adapts the pretrained tensors to this config rather than dropping them: the
    # tokenizer is restricted to input_channels' pretrained planes and to the trailing
    # frame, and the spectral filters to the smaller token grid. Anything it cannot adapt
    # raises, so a partly-random backbone cannot masquerade as a fine-tune.
    #
    # Called BEFORE apply_peft_lora: PEFT renames the parameters it wraps
    # (patch_embed.modules_to_save.default.proj.weight), and the checkpoint's keys would no
    # longer match.
    load_pretrained_weights(
        model,
        cfg.model.pretrained_path,
        channel_indices=cfg.data.input_channel_indices,
        ckpt_in_chans=len(cfg.data.pretrained_channel_order),
    )

    # Three fine-tuning regimes, selected from the model: section of the YAML:
    #   use_lora: true                          -> LoRA adapters + the head + the tokenizer
    #   use_lora: false, freeze_backbone: true  -> linear probe (head only)
    #   use_lora: false, freeze_backbone: false -> full fine-tuning
    #
    # freeze_backbone is ignored when use_lora is true: PEFT freezes every parameter, then
    # re-enables the adapters, every head_* module, and whatever
    # model.trainable_backbone_modules names.
    # Resolved up front, so a typo in the list is caught in every regime rather than only
    # when use_lora happens to be true.
    trainable_backbone = resolve_trainable_backbone_modules(
        model, cfg.model.trainable_backbone_modules
    )

    if cfg.model.freeze_backbone:
        for name, param in model.named_parameters():
            if name.startswith("backbone."):
                param.requires_grad = False
        # Re-enable what the config asked to keep trainable. Without this the linear-probe
        # regime freezes the channel-sliced tokenizer, which is the one thing this app must
        # not do: the slice is not the function the backbone was trained to consume, and no
        # other layer can rescale the tokenizer's output. The regime is then a probe of the
        # *backbone* with a trainable input layer, which is the intended reading.
        for keep in trainable_backbone:
            for param in model.get_submodule(keep).parameters():
                param.requires_grad = True

    if cfg.model.use_lora:
        model = apply_peft_lora(
            model, cfg.model.lora_config, trainable_backbone_modules=trainable_backbone
        )

    _log_trainable_parameters(model)
    return Euv2MagFinetuneLightningModule(
        model,
        metrics,
        lr=cfg.learning_rate,
        batch_size=cfg.batch_size,
        head_lr_multiplier=cfg.head_lr_multiplier,
        weight_decay=cfg.weight_decay,
    )


def _log_trainable_parameters(model) -> None:
    """Print trainable/total parameter counts, so the chosen regime is visible in the log."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    print(f"[MODEL] Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")


def build_trainer(cfg: TrainingConfig, no_wandb: bool = False) -> Tuple[L.Trainer, ModelCheckpoint]:
    """Configure loggers, callbacks, and the Lightning Trainer."""
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
        max_epochs=cfg.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="auto",
        # 32-true on CPU: neither bf16 nor fp16 autocast is useful there, and Lightning
        # warns or errors depending on the version.
        precision=cfg.precision if torch.cuda.is_available() else "32-true",
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        deterministic=cfg.deterministic,
        benchmark=False,
        logger=loggers,
        callbacks=[checkpoint_cb, upload_cb],
        log_every_n_steps=2,
    )
    return trainer, checkpoint_cb


# ---------------------------------------------------------------------------
# Evaluation figure
# ---------------------------------------------------------------------------

def save_prediction_figure(
    cfg: TrainingConfig,
    lit_model: L.LightningModule,
    val_loader: DataLoader,
    out_path: Path,
) -> None:
    """Plot truth, prediction and residual for one validation sample, in physical units.

    **The inverse transform is the point of this figure.** The model works in normalized
    space, where an MSE is in units of a per-channel standard deviation and means nothing
    physically. ``dataset.inverse_transform_data`` undoes *both* normalization stages --
    the z-score and the signum-log -- returning Gauss, which is the only space in which
    "is this magnetogram right" is a meaningful question. See the THE THREE SPACES block
    at the top of ``workshop_infrastructure/datasets/helio.py``; using
    ``scaler.inverse_transform()`` here instead would leave the values in signum-log space
    and the colour bar would be wrong by orders of magnitude.
    """
    import matplotlib
    matplotlib.use("Agg")  # No display on a compute node.
    import matplotlib.pyplot as plt
    import numpy as np

    dataset = val_loader.dataset
    lit_model.eval()
    batch = next(iter(val_loader))

    device = next(lit_model.parameters()).device
    on_device = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        prediction = lit_model(on_device)

    # (B, C, H, W) -> the first sample's channel stack, as float32 on the CPU. float() is
    # needed because bf16-mixed leaves the output in bfloat16, which numpy cannot take.
    pred = prediction[0].float().cpu().numpy()
    truth = batch["forecast"][0].squeeze(1).float().cpu().numpy()

    # inverse_transform_subset, not inverse_transform_data: the latter indexes its
    # per-channel statistics positionally by dataset.channels, and these stacks hold
    # target_channels in their own order. See the method's docstring -- getting this wrong
    # does not raise, it silently returns the wrong channel's units.
    pred_phys = dataset.inverse_transform_subset(pred, dataset.target_channels)
    truth_phys = dataset.inverse_transform_subset(truth, dataset.target_channels)

    n = len(dataset.target_channels)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.2 * n), squeeze=False)
    for row, name in enumerate(dataset.target_channels):
        t, p = truth_phys[row], pred_phys[row]
        # Symmetric limits from the truth, shared by both panels: a magnetogram is signed,
        # and independently scaled panels would make a flat prediction look structured.
        lim = float(np.percentile(np.abs(t), 99.5)) or 1.0
        for col, (img, title, kw) in enumerate([
            (t, f"{name} truth", dict(cmap="gray", vmin=-lim, vmax=lim)),
            (p, f"{name} prediction", dict(cmap="gray", vmin=-lim, vmax=lim)),
            (p - t, f"{name} residual (pred - truth)", dict(cmap="coolwarm", vmin=-lim, vmax=lim)),
        ]):
            ax = axes[row][col]
            im = ax.imshow(img, origin="lower", **kw)
            ax.set_title(title, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, label="Gauss")

    fig.suptitle(
        f"{cfg.job_id} — {'+'.join(cfg.data.input_channels)} -> "
        f"{'+'.join(cfg.data.target_channels)}, {cfg.model.img_size}px",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] Wrote {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("medium")

    cfg = load_euv2mag_config(args.config)
    # Seeding comes after the config load, so the seed is a configured value rather than a
    # constant buried in the code.
    L.seed_everything(cfg.seed, workers=True)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.max_train_samples is not None:
        cfg.data.max_train_samples = args.max_train_samples
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    if args.s3_cache_dir is not None:
        cfg.data.s3_cache_dir = args.s3_cache_dir
    if args.deterministic is not None:
        cfg.deterministic = _DETERMINISTIC_CLI[args.deterministic]
    ensure_assets(cfg, which=["scalers"] if args.train_baseline else ["scalers", "weights"])

    scalers = build_scalers(info=cfg.data.scalers_path)
    train_loader, val_loader = build_datasets(cfg, scalers)
    lit_model = build_model(cfg, train_baseline=args.train_baseline)
    print(
        f"[DATA] train: {len(train_loader.dataset)} samples | "
        f"val: {len(val_loader.dataset)} samples | "
        f"batch_size: {cfg.batch_size} x {cfg.accumulate_grad_batches} accumulated | "
        f"frames: {cfg.model.img_size}x{cfg.model.img_size} (data.pooling={cfg.data.pooling}) | "
        f"{'+'.join(cfg.data.input_channels)} -> {'+'.join(cfg.data.target_channels)}"
    )
    trainer, checkpoint_cb = build_trainer(cfg, no_wandb=args.no_wandb)

    trainer.fit(lit_model, train_loader, val_loader)

    if checkpoint_cb.best_model_path:
        print(f"[CKPT] Best checkpoint: {checkpoint_cb.best_model_path}")
        if checkpoint_cb.best_model_score is not None:
            print(f"[CKPT] Best val_loss: {float(checkpoint_cb.best_model_score):.6f}")
    else:
        print("[CKPT] No best checkpoint was saved.")

    # Rank 0 only: under DDP every rank would otherwise write the same file.
    if not args.no_figure and trainer.is_global_zero:
        save_prediction_figure(
            cfg, lit_model, val_loader, Path(cfg.output.ckpt_dir) / f"{cfg.job_id}_prediction.png"
        )


if __name__ == "__main__":
    main()
