#!/usr/bin/env python3
"""
Runnable finetuning script derived from `2_finetune_template_1D.ipynb`.

Design goals
- Minimal CLI arguments
- Multi-GPU capable (DDP) when run as a script
- Mirrors notebook behavior closely

Assumptions
- You have already downloaded `scalers.yaml` + model weights (the notebook ran `download_scalers_and_weights.sh`).
- You run this script from the main path and specify your devices in the command line along the python call
    (e.g., `CUDA_VISIBLE_DEVICES=0,1 python -m downstream_apps.template.3_finetune_template_1D ...`).

Usage
  CUDA_VISIBLE_DEVICES=6,7 python -m downstream_apps.template.3_finetune_template_1D --config /home/amjlowlevel/surya_workshop/downstream_apps/template/configs/config_script.yaml --batch-size 2 --max-epochs 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Union

import torch
import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size under DDP.")
    parser.add_argument("--no-wandb", action="store_true")    
    parser.add_argument("--train_baseline", action="store_true")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    # Mirror notebook sys.path adjustments
    script_dir = Path(__file__).resolve().parent
    sys.path.append(str((script_dir / "../../").resolve()))
    sys.path.append(str((script_dir / "../../Surya").resolve()))

    # Determinism similar to typical Lightning usage
    L.seed_everything(42, workers=True)

    # ---------------------------------------------------------------------
    # Config + scalers
    # ---------------------------------------------------------------------
    config = load_yaml(args.config)
    config["data"]["scalers"] = load_yaml(config["data"]["scalers_path"])

    from surya.utils.data import build_scalers
    scalers = build_scalers(info=config["data"]["scalers"])

    # ---------------------------------------------------------------------
    # Dataset + loaders
    # ---------------------------------------------------------------------
    from downstream_apps.template.datasets.template_dataset import FlareDSDataset

    common_ds_kwargs = dict(
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        n_input_timestamps=config["model"]["time_embedding"]["time_dim"],
        rollout_steps=config["rollout_steps"],
        channels=config["data"]["channels"],
        drop_hmi_probability=config["drop_hmi_probability"],
        use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
        scalers=scalers,
        s3_use_simplecache=False,
        s3_cache_dir="/tmp/helio_s3_cache",
        # Downstream-specific
        return_surya_stack=True,
        max_number_of_samples=10,
        ds_flare_index_path=config["data"]["flare_index_path"],
        ds_time_column="start_time",
        ds_time_tolerance="4d",
        ds_match_direction="forward",
    )

    train_dataset = FlareDSDataset(
        index_path=config["data"]["train_data_path"],
        phase="train",
        **common_ds_kwargs,
    )
    val_dataset = FlareDSDataset(
        index_path=config["data"]["valid_data_path"],
        phase="val",
        **common_ds_kwargs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        multiprocessing_context="spawn",
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        multiprocessing_context="spawn",
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )

    # ---------------------------------------------------------------------
    # Model + PEFT (as in notebook)
    # ---------------------------------------------------------------------
    from downstream_apps.template.metrics.template_metrics import FlareMetrics
    from downstream_apps.template.lightning_modules.pl_simple_baseline import FlareLightningModule

    if args.train_baseline:
        from downstream_apps.template.models.simple_baseline import RegressionFlareModel
        n_input_timestamps = config["model"]["time_embedding"]["time_dim"]
        n_channels = len(config["data"]["channels"])

        model = RegressionFlareModel(n_input_timestamps*n_channels, config["data"]["channels"], scalers)        

    else:

        from workshop_infrastructure.utils import apply_peft_lora
        from workshop_infrastructure.models.finetune_models import HelioSpectformer1D

        model = HelioSpectformer1D(
            img_size=config["model"]["img_size"],
            patch_size=config["model"]["patch_size"],
            in_chans=config["model"]["in_channels"],
            embed_dim=config["model"]["embed_dim"],
            time_embedding=config["model"]["time_embedding"],
            depth=config["model"]["depth"],
            num_heads=config["model"]["num_heads"],
            mlp_ratio=config["model"]["mlp_ratio"],
            drop_rate=config["model"]["drop_rate"],
            dtype=config["dtype"],
            window_size=config["model"]["window_size"],
            dp_rank=config["model"]["dp_rank"],
            learned_flow=config["model"]["learned_flow"],
            use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
            init_weights=config["model"]["init_weights"],
            checkpoint_layers=config["model"]["checkpoint_layers"],
            n_spectral_blocks=config["model"]["spectral_blocks"],
            rpe=config["model"]["rpe"],
            ensemble=config["model"]["ensemble"],
            finetune=config["model"]["finetune"],
            nglo=config["model"]["nglo"],
            # Finetuning additions
            dropout=config["model"]["dropout"],
            num_penultimate_transformer_layers=0,
            num_penultimate_heads=0,
            num_outputs=1,
            config=config,
        )

        # Load pretrained weights if provided
        pretrained_path = config.get("pretrained_path")
        if pretrained_path:
            print(f"Loading pretrained model from {pretrained_path}.")
            model_state = model.state_dict()
            checkpoint_state = torch.load(pretrained_path, weights_only=True, map_location="cpu")
            filtered_checkpoint_state = {
                k: v
                for k, v in checkpoint_state.items()
                if k in model_state and hasattr(v, "shape") and v.shape == model_state[k].shape
            }
            model_state.update(filtered_checkpoint_state)
            model.load_state_dict(model_state, strict=True)

        # Optional: apply LoRA via config (mirrors notebook intent)
        model = apply_peft_lora(model, config)


    # Metrics + LightningModule
    metrics = {
        "train_loss": FlareMetrics("train_loss"),
        "train_metrics": FlareMetrics("train_metrics"),
        "val_metrics": FlareMetrics("val_metrics"),
    }

    # The notebook uses a simple baseline LightningModule wrapper
    lit_model = FlareLightningModule(model, metrics, lr=config.get("learning_rate", 1e-3), batch_size=args.batch_size)

    # ---------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------
    loggers = []
    if not args.no_wandb:
        # Notebook values
        project_name = "template_flare_regression"
        run_name = "baseline_experiment_1"
        wandb_logger = WandbLogger(
            entity="surya_handson",
            project=project_name,
            name=run_name,
            log_model=False,
            save_dir=os.environ.get("TMPDIR", "./wandb/wandb_tmp"),
        )
        loggers.append(wandb_logger)

    loggers.append(CSVLogger("runs", name="simple_flare"))

    # ---------------------------------------------------------------------
    # Trainer (multi-GPU ready)
    # ---------------------------------------------------------------------

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        logger=loggers,
        callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)],
        log_every_n_steps=2,
    )

    trainer.fit(lit_model, train_loader, val_loader)


if __name__ == "__main__":
    main()
