import os
import sys
import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, Callable, Tuple, Any, Optional

from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
import wandb

# --- SETUP PATHS ---
sys.path.append("../../")
sys.path.append("../../Surya")

from surya.utils.data import build_scalers
from workshop_infrastructure.utils import apply_peft_lora
from workshop_infrastructure.models.finetune_models import HelioSpectformer1D
from downstream_apps.template.lightning_modules.pl_simple_baseline import FlareLightningModule
from downstream_apps.andong.datasets.dataset_andong import DstDataset

# ==========================================
#  USER CONTROL FLAGS
# ==========================================
# True  = Run Training Loop -> Then Visualize
# False = Skip Training -> Just Visualize
TRAIN_FLAG = True  

# Checkpoint path for Visualization
# (If TRAIN_FLAG=True, this will be overwritten by the new best model path)
CKPT_PATH = "runs/dst_forecast/dst_finetune_3day_delay_multiGPU/checkpoints/best.ckpt"

# Visualization Settings
VIZ_START_DATE = "2024-05-10 00:00"  # Approximate start for plotting time axis
# ==========================================


# --- 1. CONFIGURATION ---
def setup_environment():
    # Memory Optimizations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.set_float32_matmul_precision('medium')
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2" 

    # WandB Setup
    if "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_API_KEY"] = "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"

    os.environ["WANDB_DIR"] = "./wandb/wandb_logs"
    os.environ["WANDB_CACHE_DIR"] = "./wandb/wandb_cache"
    os.environ["WANDB_CONFIG_DIR"] = "./wandb/wandb_config"
    
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
    os.makedirs(os.environ["WANDB_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["WANDB_CONFIG_DIR"], exist_ok=True)
    
    # try:
    #     wandb.login()
    # except:
    #     print("⚠️ WandB login failed. Logs will be local only.")


class DstMetrics:
    def __init__(self, mode="train_loss"):
        self.mode = mode
        
    def __call__(self, preds, target):
        if target.dim() == 3: target = target[:, 0, :]
        if preds.dim() == 3 and preds.shape[-1] == 1: preds = preds.squeeze(-1)
        loss = torch.nn.functional.mse_loss(preds, target)
        
        if self.mode == "train_loss":
            return {"mse": loss}, [1]
        else:
            return {"mse": loss}, []


# --- 2. VISUALIZATION FUNCTIONS ---

def load_model_structure(config):
    """Initializes the model structure."""
    return HelioSpectformer1D(
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
        dropout=config["model"]["dropout"],
        num_penultimate_transformer_layers=0,
        num_penultimate_heads=0,
        num_outputs=216,
        config=config,
    )

def visualize_event(config, scalers, dst_data_path, cache_dir, checkpoint_path):
    """Loads checkpoint, runs inference, and plots with Time Axis."""
    print(f"\n🎨 Starting Visualization Mode...")
    print(f"   Loading Checkpoint: {checkpoint_path}")
    
    # 1. Setup Validation Dataset (No Filters)
    dataset = DstDataset(
        index_path=config["data"]["valid_data_path"], 
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        n_input_timestamps=config["model"]["time_embedding"]["time_dim"],
        rollout_steps=config["rollout_steps"],
        channels=config["data"]["channels"],
        drop_hmi_probability=0.0, 
        use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
        scalers=scalers,
        phase="val",
        s3_use_simplecache=False,
        s3_cache_dir=None,
        # s3_cache_dir=cache_dir,
        return_surya_stack=True,
        dst_hdf5_path=dst_data_path,
        delay_days=3,
        max_number_of_samples=None,
        storm_threshold=None # See all data
    )
    
    # 2. Load Model
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found at: {checkpoint_path}")
        return

    model = load_model_structure(config)
    model = apply_peft_lora(model, config)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle Lightning State Dict
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v 
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # 3. Run Inference
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    predictions, actuals = [], []
    timestamps = []
    
    # Generate mock timestamps if dataset doesn't provide them
    # Assuming 3-hour cadence based on dataset name
    start_time = pd.Timestamp(VIZ_START_DATE)
    
    print("   Running inference on validation set (Limit: 300 samples)...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 300: break 
            
            # Handle Inputs
            if isinstance(batch, dict) and "images" in batch:
                img_stack = batch["images"].to(device)
            else:
                img_stack = batch.to(device)
            
            target = batch["forecast"]
            
            # Predict
            pred = model(img_stack)
            
            # Store Data
            predictions.append(pred[0, 0].item()) 
            actuals.append(target[0, 0].item() if target.dim() < 3 else target[0, 0, 0].item())
            
            # Generate Time (Add 3 hours per step)
            # If your dataset has real times, extract them here: e.g. batch['time'][0]
            current_time = start_time + timedelta(hours=3 * i)
            timestamps.append(current_time)

            if i % 50 == 0:
                print(f"   Processed {i} samples...")

    # 4. Plot with Time Axis
    plt.figure(figsize=(12, 6))
    
    plt.plot(timestamps, actuals, label="Actual Dst", color="black", linewidth=2)
    plt.plot(timestamps, predictions, label="Predicted Dst", color="red", linestyle="--")
    
    plt.title(f"Dst Forecast Visualization (Starting {VIZ_START_DATE})")
    plt.ylabel("Dst (nT)")
    plt.xlabel("Time")
    
    # Format X-Axis as Dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%h'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.gcf().autofmt_xdate() # Rotate dates
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = "event_prediction_time_series.png"
    plt.savefig(out_file)
    print(f"✅ Plot saved to {out_file}")


# --- 3. MAIN EXECUTION ---
def main():
    setup_environment()
    L.seed_everything(42, workers=True)

    print(f"📋 Loading configuration...")
    config_path = "./configs/config.yaml"
    config = yaml.safe_load(open(config_path, "r"))
    config["data"]["scalers"] = yaml.safe_load(open(config["data"]["scalers_path"], "r"))
    scalers = build_scalers(info=config["data"]["scalers"])

    dst_data_path = "/media/faraday/andong/Dataspace/GONG_NN/Data/ML_Ready_Dataset_2019-2026-3h.h5"
    CACHE_DIR = "/media/faraday/andong/Workspace/surya_workshop/cache"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Global variable to store the path of the model we want to visualize
    target_checkpoint = CKPT_PATH

    # ==========================================
    # STEP 1: TRAINING (Optional)
    # ==========================================
    if TRAIN_FLAG:
        print("🚀 Starting Training Mode...")
        
        train_dataset = DstDataset(
            index_path=config["data"]["train_data_path"],
            time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
            time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
            n_input_timestamps=config["model"]["time_embedding"]["time_dim"],
            rollout_steps=config["rollout_steps"],
            channels=config["data"]["channels"],
            drop_hmi_probability=config["drop_hmi_probability"],
            use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
            scalers=scalers,
            phase="train",
            s3_use_simplecache=False,
            s3_cache_dir=None,
            # s3_cache_dir=CACHE_DIR,
            return_surya_stack=True,
            dst_hdf5_path=dst_data_path,
            delay_days=3,
            max_number_of_samples=None,
            storm_threshold=-350.0  # Training on Extreme Storms
        )

        val_dataset = DstDataset(
            index_path=config["data"]["valid_data_path"],
            time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
            time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
            n_input_timestamps=config["model"]["time_embedding"]["time_dim"],
            rollout_steps=config["rollout_steps"],
            channels=config["data"]["channels"],
            drop_hmi_probability=config["drop_hmi_probability"],
            use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
            scalers=scalers,
            phase="val",
            s3_use_simplecache=False,
            s3_cache_dir=None,
            # s3_cache_dir=CACHE_DIR,
            return_surya_stack=True,
            dst_hdf5_path=dst_data_path,
            delay_days=3,
            max_number_of_samples=None,
            storm_threshold=-60  # Validate on ALL data
        )

        batch_size = 1
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, persistent_workers=False)

        model = load_model_structure(config)
        
        print("Loading Pretrained Weights...")
        checkpoint_state = torch.load(config["pretrained_path"], weights_only=True, map_location="cpu")
        model_state = model.state_dict()
        filtered_checkpoint_state = {k: v for k, v in checkpoint_state.items() if k in model_state and v.shape == model_state[k].shape}
        model_state.update(filtered_checkpoint_state)
        model.load_state_dict(model_state, strict=True)

        print("Applying LoRA...")
        model = apply_peft_lora(model, config)

        metrics = {
            'train_loss': DstMetrics("train_loss"),
            'train_metrics': DstMetrics("train_metrics"),
            'val_metrics': DstMetrics("val_metrics")
        }
        
        lit_model = FlareLightningModule(model, metrics, lr=1e-5, batch_size=batch_size)

        wandb_logger = WandbLogger(entity="surya_handson", project="surya_dst_forecast", name="dst_finetune_v3", save_dir=os.environ["WANDB_DIR"])
        csv_logger = CSVLogger("runs", name="dst_forecast_new")

        # Save Best Model Callback
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best_model" # Explicit name makes it easier to find
            save_last=True,         # <--- Recommended: Always saves "last.ckpt" even if val_loss is weird
        )

        trainer = L.Trainer(
            max_epochs=30,
            accelerator="auto",
            devices="auto",
            strategy="ddp",
            precision="bf16-mixed",
            logger=[wandb_logger, csv_logger],
            callbacks=[checkpoint_callback],
            accumulate_grad_batches=8,
            log_every_n_steps=2,
            num_sanity_val_steps=0, # Skip sanity check to see if training starts
        )

        trainer.fit(lit_model, train_loader, val_loader)
        
        # UPDATE target checkpoint to the one we just trained
        target_checkpoint = checkpoint_callback.best_model_path
        print(f"\n✅ Training Complete. Best model saved at: {target_checkpoint}")

    # ==========================================
    # STEP 2: ALWAYS VISUALIZE (Uses target_checkpoint)
    # ==========================================
    # In main(), wrap the visualization call:
    if trainer.global_rank == 0:
        if target_checkpoint and os.path.exists(target_checkpoint):
            visualize_event(config, scalers, dst_data_path, CACHE_DIR, target_checkpoint)
        else:
            # Fallback if training didn't produce a file (e.g. crash) or path is wrong
            print(f"⚠️ Could not find checkpoint at {target_checkpoint}. Using default fallback: {CKPT_PATH}")
            visualize_event(config, scalers, dst_data_path, CACHE_DIR, CKPT_PATH)

if __name__ == "__main__":
    main()