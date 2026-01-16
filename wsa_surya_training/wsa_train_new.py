#!/usr/bin/env python3
"""
wsa_train.py

Main training script for WSA (Wang-Sheeley-Arge) map prediction.

This script orchestrates the complete training pipeline following the workshop template:
  - Loads configuration from wsa_config.yaml
  - Creates train/val/test datasets with specified CR lists
  - Loads pre-trained Surya encoder using workshop_infrastructure
  - Applies LoRA (Low-Rank Adaptation) for efficient fine-tuning (optional)
  - Sets up PyTorch Lightning training with logging and checkpointing
  - Trains the model on 2D WSA map prediction task

Usage:
    python wsa_train.py
    
    Optional CLI arguments:
    python wsa_train.py --config ./configs/wsa_config.yaml --devices 1 --use-lora
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

import torch
import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

#import multiprocessing
#multiprocessing.set_start_method('spawn', force=True)

# ============================================================================
# Path Resolution (Following Workshop Template Pattern)
# ============================================================================

# Determine the absolute path to the script's directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Determine the absolute path to the main project root
PROJECT_ROOT = (SCRIPT_DIR / "../").resolve()

# Construct absolute paths to Surya directory
SURYA_DIR = PROJECT_ROOT / "./Surya"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SURYA_DIR) not in sys.path:
    sys.path.insert(0, str(SURYA_DIR))

# ============================================================================
# Utilities
# ============================================================================

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_wandb_dirs() -> None:
    """Ensure W&B directories exist and are writable."""
    os.environ.setdefault("WANDB_DIR", "./wandb/wandb_logs")
    os.environ.setdefault("WANDB_CACHE_DIR", "./wandb/wandb_cache")
    os.environ.setdefault("WANDB_CONFIG_DIR", "./wandb/wandb_config")
    os.environ.setdefault("TMPDIR", "./wandb/wandb_tmp")

    for k in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "TMPDIR"):
        Path(os.environ[k]).mkdir(parents=True, exist_ok=True)


def parse_devices_arg(dev: str) -> Union[str, int, list]:
    """
    Accept:
      --devices auto
      --devices 1
      --devices 2
      --devices 0,1,2,3
    """
    if dev == "auto":
        return "auto"
    if "," in dev:
        return [int(x.strip()) for x in dev.split(",") if x.strip()]
    return int(dev)


def infer_strategy(devices) -> str:
    """Infer training strategy based on number of devices."""
    if devices == "auto":
        return "auto"
    if isinstance(devices, int):
        return "ddp" if devices > 1 else "auto"
    if isinstance(devices, list):
        return "ddp" if len(devices) > 1 else "auto"
    return "auto"


# ============================================================================
# Main Script
# ============================================================================

def main() -> None:
    """Main training orchestration for 2D WSA map prediction."""
    print("=" * 80)
    print("WSA Map Prediction Training Script (2D)")
    print("=" * 80)

    # ========================================================================
    # Parse Arguments
    # ========================================================================
    parser = argparse.ArgumentParser(
        description="Train WSA 2D map prediction model with optional LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wsa_train.py
  python wsa_train.py --config ./configs/wsa_config.yaml --devices 1
  python wsa_train.py --use-lora --unfreeze-encoder --batch-size 4
  python wsa_train.py --devices 0,1,2,3 --max-epochs 20
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/wsa_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="3",
        help='GPU devices: "auto", "1", "2", or "0,1,2"',
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max epochs from config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA fine-tuning (overrides config)",
    )
    parser.add_argument(
        "--unfreeze-encoder",
        action="store_true",
        help="Unfreeze encoder for full fine-tuning (default: frozen)",
    )
    args = parser.parse_args()

    # ========================================================================
    # Setup Environment
    # ========================================================================
    torch.set_float32_matmul_precision("medium")
    ensure_wandb_dirs()
    os.environ["TMPDIR"] = "/tmp"
    L.seed_everything(42, workers=True)

    print(f"\n✅ Environment configured")
    print(f"   Script dir: {SCRIPT_DIR}")
    print(f"   Project root: {PROJECT_ROOT}")

    # ========================================================================
    # Load Configuration
    # ========================================================================
    print(f"\n📋 Loading configuration from {args.config}...")
    
    config_path = SCRIPT_DIR / args.config
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_yaml(config_path)
    
    from pathlib import Path

    # Load scalers relative to config directory
    scalers_path = Path(config["data"]["scalers_path"])  # Convert to Path object
    if not scalers_path.exists():
        print(f"❌ Scalers file not found: {scalers_path}")
        raise FileNotFoundError(f"Scalers file not found: {scalers_path}")

    config["data"]["scalers_path"] = load_yaml(scalers_path)
    
    print("✅ Configuration loaded successfully!")

    # Override from CLI if provided
    if args.max_epochs:
        config["optimizer"]["max_epochs"] = args.max_epochs
        print(f"   ℹ️  Overriding max_epochs to {args.max_epochs}")
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
        print(f"   ℹ️  Overriding batch_size to {args.batch_size}")
    if args.use_lora:
        config["use_lora"] = True
        print(f"   ℹ️  Enabling LoRA via CLI flag")

    # ========================================================================
    # Build Scalers
    # ========================================================================
    print(f"\n📊 Building scalers...")
    from surya.utils.data import build_scalers
    scalers = build_scalers(info=config["data"]["scalers_path"])
    print("✅ Scalers built successfully!")

    # ========================================================================
    # Create Datasets & DataLoaders
    # ========================================================================
    print(f"\n📊 Creating datasets...")

    from datasets.wsa_dataset import WSAImageDataset

    dataset_kwargs = {
        "surya_index_path": config["data"]["surya_index_path"],
        "wsa_map_dir": config["data"]["wsa_map_dir"],
        "wsa_params_path": config["data"]["wsa_params_path"],
        "scalers": scalers,
        "channels": config["data"]["channels"],
        "normalize_wsa": config.get("normalize_wsa_maps", True),
        "tolerance_days": config["data"]["cr_tolerance_days"],
        "s3_use_simplecache": config["data"]["s3_use_simplecache"],
        "s3_cache_dir": config["data"]["s3_cache_dir"],
    }

    train_dataset = WSAImageDataset(
        cr_list=config["data"]["train_crs"],
        phase="train",
        **dataset_kwargs,
    )
    val_dataset = WSAImageDataset(
        cr_list=config["data"]["val_crs"],
        phase="val",
        **dataset_kwargs,
    )
    test_dataset = WSAImageDataset(
        cr_list=config["data"]["test_crs"],
        phase="test",
        **dataset_kwargs,
    )

    print(f"✅ Datasets created:")
    print(f"   - Train: {len(train_dataset)} samples")
    print(f"   - Val:   {len(val_dataset)} samples")
    print(f"   - Test:  {len(test_dataset)} samples")

    # ========================================================================
    # Create DataLoaders
    # ========================================================================
    print(f"\n🔄 Creating dataloaders...")

    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_data_workers"]

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"✅ Dataloaders created (batch_size: {batch_size}, num_workers: {num_workers})")

    # ========================================================================
    # Load Encoder (Using workshop_infrastructure)
    # ========================================================================
    print(f"\n🧠 Loading Surya encoder from workshop_infrastructure...")
    
    from workshop_infrastructure.models.finetune_models import HelioSpectformer2D
    
    try:
        model = HelioSpectformer2D(
            img_size=config["model"]["img_size"],
            patch_size=config["model"]["patch_size"],
            in_chans=config["model"]["in_chans"],
            embed_dim=config["model"]["embed_dim"],
            time_embedding=config["model"]["time_embedding"],
            depth=config["model"]["depth"],
            num_heads=config["model"]["num_heads"],
            mlp_ratio=config["model"]["mlp_ratio"],
            drop_rate=config["model"]["drop_rate"],
            dtype=torch.bfloat16,  # ← Fixed dtype
            window_size=config["model"]["window_size"],
            dp_rank=config["model"]["dp_rank"],
            learned_flow=config["model"]["learned_flow"],
            use_latitude_in_learned_flow=config["model"].get("use_latitude_in_learned_flow", False),
            init_weights=config["model"]["init_weights"],
            checkpoint_layers=config["model"]["checkpoint_layers"],
            n_spectral_blocks=config["model"]["spectral_blocks"],
            rpe=config["model"]["rpe"],
            finetune=config["model"]["finetune"],
            config=config,  # ← REQUIRED!
        )

        # Load pretrained weights if provided
        pretrained_path = config.get("pretrained_path") or config["model"].get("checkpoint_path")
        if pretrained_path:
            # ✅ RESOLVE RELATIVE PATHS from config directory
            if not Path(pretrained_path).is_absolute():
                pretrained_path = (config_path.parent / pretrained_path).resolve()
            
            pretrained_path = str(pretrained_path)  # Convert to string for torch.load
            
            print(f"Loading pretrained model from {pretrained_path}.")
            
            if not Path(pretrained_path).exists():
                print(f"⚠️  Checkpoint not found: {pretrained_path}")
                print(f"   Skipping pretrained weights, using random initialization")
            else:
                model_state = model.state_dict()
                checkpoint_state = torch.load(pretrained_path, weights_only=True, map_location="cpu")
                filtered_checkpoint_state = {
                    k: v
                    for k, v in checkpoint_state.items()
                    if k in model_state and hasattr(v, "shape") and v.shape == model_state[k].shape
                }
                model_state.update(filtered_checkpoint_state)
                model.load_state_dict(model_state, strict=True)
                print("✅ Pretrained weights loaded successfully!")
        else:
            print(f"⚠️  No pretrained_path or checkpoint_path in config, using random initialization")
        
    except Exception as e:
        print(f"❌ Error loading encoder: {e}")
        import traceback
        traceback.print_exc()
        print("   Falling back to randomly initialized encoder...")
        
        # Fallback: create model with minimal required params
        model = HelioSpectformer2D(
            img_size=config["model"]["img_size"],
            patch_size=config["model"]["patch_size"],
            in_chans=config["model"]["in_chans"],
            embed_dim=config["model"]["embed_dim"],
            time_embedding=config["model"]["time_embedding"],
            depth=config["model"]["depth"],
            num_heads=config["model"]["num_heads"],
            mlp_ratio=config["model"]["mlp_ratio"],
            drop_rate=config["model"]["drop_rate"],
            window_size=config["model"]["window_size"],
            dp_rank=config["model"]["dp_rank"],
            learned_flow=config["model"]["learned_flow"],
            use_latitude_in_learned_flow=config["model"].get("use_latitude_in_learned_flow", False),
            init_weights=config["model"]["init_weights"],
            checkpoint_layers=config["model"]["checkpoint_layers"],
            n_spectral_blocks=config["model"]["spectral_blocks"],
            rpe=config["model"]["rpe"],
            finetune=config["model"]["finetune"],
            config=config,  # ← REQUIRED!
        ).cuda()
        print("✅ Randomly initialized encoder ready!")

    # ========================================================================
    # Apply LoRA (Using workshop_infrastructure utility)
    # ========================================================================
    use_lora = config.get("use_lora", False)
    
    if use_lora:
        print(f"\n🔧 Applying LoRA fine-tuning from workshop_infrastructure...")
        from workshop_infrastructure.utils import apply_peft_lora
        model = apply_peft_lora(model, config)
        print("✅ LoRA applied successfully!")
    else:
        print(f"\n⏭️  LoRA disabled (use --use-lora to enable)")

    # ========================================================================
    # Create Metrics (as instances, following template pattern)
    # ========================================================================
    print(f"\n📊 Creating metrics...")
    
    from metrics.wsa_metrics import WSAMetrics
    
    metrics = {
        "train_loss": WSAMetrics("train_loss"),
        "train_metrics": WSAMetrics("train_metrics"),
        "val_metrics": WSAMetrics("val_metrics"),
    }
    
    print("✅ Metrics created (MAE loss and evaluation)")

    # ========================================================================
    # Create Lightning Module
    # ========================================================================
    print(f"\n⚡ Creating Lightning module...")
    
    from lightning_modules.wsa_lightning_module import WSALightningModule
    
    lit_model = WSALightningModule(
        model=model,
        metrics=metrics,
        lr=config["optimizer"]["learning_rate"],
        batch_size=config["data"]["batch_size"],
    )
    
    print("✅ Lightning module created")

    # ========================================================================
    # Setup Loggers
    # ========================================================================
    print(f"\n📝 Setting up loggers...")
    
    loggers = []
    
    if not args.no_wandb:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config['wandb']['run_name']}_{timestamp}"
        
        wandb_logger = WandbLogger(
            entity=config["wandb"]["entity"],
            project=config["wandb"]["project"],
            name=run_name,
            log_model=False,
            save_dir=os.environ.get("WANDB_DIR", "./wandb/wandb_logs"),
        )
        loggers.append(wandb_logger)
        print(f"✅ W&B Logger: {config['wandb']['project']} / {run_name}")
    else:
        print("⏭️  W&B logging disabled (--no-wandb)")
    
    csv_logger = CSVLogger("runs", name="wsa_training")
    loggers.append(csv_logger)
    print(f"✅ CSV Logger: runs/wsa_training/")

    # ========================================================================
    # Setup Checkpointing
    # ========================================================================
    checkpoint_dir = Path(config["checkpoint"]["save_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch_{epoch:02d}-val_loss_{val_loss:.4f}",
        monitor=config["checkpoint"]["monitor"],
        mode=config["checkpoint"]["mode"],
        save_top_k=config["checkpoint"]["save_top_k"],
        verbose=True,
    )
    print(f"✅ Checkpointing: {checkpoint_dir}")

    # ========================================================================
    # Setup Trainer (Multi-GPU Ready)
    # ========================================================================
    print(f"\n🚀 Setting up trainer...")
    
    devices = parse_devices_arg(args.devices)
    strategy = infer_strategy(devices)
    
    trainer = L.Trainer(
        max_epochs=config["optimizer"]["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        strategy=strategy,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        logger=loggers,
        callbacks=[checkpoint_callback],
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        num_sanity_val_steps=0,  # Skip sanity check
    )
    
    print(f"✅ Trainer configured:")
    print(f"   - Devices: {devices}")
    print(f"   - Strategy: {strategy}")
    print(f"   - Max Epochs: {config['optimizer']['max_epochs']}")
    print(f"   - Precision: bf16-mixed")

    # ========================================================================
    # Training Pipeline Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training Pipeline Summary")
    print("=" * 80)
    print(f"Task: 2D WSA Map Prediction")
    print(f"Model: HelioSpectformer2D (from workshop_infrastructure)")
    print(f"LoRA Enabled: {use_lora}")
    print(f"Learning Rate: {config['optimizer']['learning_rate']}")
    print(f"Batch Size: {batch_size}")
    print(f"Max Epochs: {config['optimizer']['max_epochs']}")
    print(f"Train CRs: {config['data']['train_crs']}")
    print(f"Val CRs: {config['data']['val_crs']}")
    print(f"Test CRs: {config['data']['test_crs']}")
    print("=" * 80 + "\n")

    # ========================================================================
    # Train
    # ========================================================================
    print("🚀 Starting training...\n")
    
    trainer.fit(lit_model, train_loader, val_loader)
    
    print("\n" + "=" * 80)
    print("Training Complete")
    print("=" * 80)
    print(f"Job ID: {config.get('job_id', 'N/A')}")
    print(f"Checkpoint Dir: {checkpoint_dir}")
    print(f"Logs Dir: runs/wsa_training/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()