import os
import sys
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- SETUP PATHS (Same as Main_dst.py) ---
sys.path.append("../../")
sys.path.append("../../Surya")

from surya.utils.data import build_scalers
from workshop_infrastructure.models.finetune_models import HelioSpectformer1D
from downstream_apps.andong.datasets.dataset_andong import DstDataset

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Path to your best checkpoint (Update this!)
CKPT_PATH = "runs/dst_forecast/dst_finetune_3day_delay_multiGPU/checkpoints/best_model.ckpt"

# 2. Path to your training log CSV (Update version number if needed)
LOG_CSV_PATH = "runs/dst_forecast_new/version_9/metrics.csv" 

# 3. Output folder for figures
FIG_DIR = "Figs/"
os.makedirs(FIG_DIR, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==========================================
# PART 1: Plot Training & Validation Loss
# ==========================================
def plot_loss_curves():
    print(f"📊 Plotting Loss Curves from {LOG_CSV_PATH}...")
    try:
        df = pd.read_csv(LOG_CSV_PATH)
        
        # Group by epoch to handle step-wise logging
        df_epoch = df.groupby("epoch").mean()

        plt.figure(figsize=(10, 6))
        
        # Plot Train Loss (if available)
        if "train_loss" in df_epoch.columns:
            plt.plot(df_epoch.index, df_epoch["train_loss"], label="Train Loss", marker='o')
            
        # Plot Val Loss (if available)
        if "val_loss" in df_epoch.columns:
            plt.plot(df_epoch.index, df_epoch["val_loss"], label="Val Loss", marker='x', linestyle='--')
            
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{FIG_DIR}/loss_curve.png", dpi=300)
        plt.close()
        print(f"   -> Saved to {FIG_DIR}/loss_curve.png")
        
    except FileNotFoundError:
        print("⚠️ metrics.csv not found. Skipping loss plot.")

# ==========================================
# PART 2: Evaluate Model on Validation Set
# ==========================================
def run_evaluation():
    print("🚀 Loading Model & Data for Evaluation...")
    
    # 1. Load Config
    config_path = "./configs/config.yaml"
    config = yaml.safe_load(open(config_path, "r"))
    config["data"]["scalers"] = yaml.safe_load(open(config["data"]["scalers_path"], "r"))
    scalers = build_scalers(info=config["data"]["scalers"])
    
    # 2. Setup Validation Dataset (Same as Main_dst.py)
    dst_data_path = "/media/faraday/andong/Dataspace/GONG_NN/Data/ML_Ready_Dataset_2019-2026-3h.h5"
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
        s3_use_simplecache=True, # Use cache for speed
        s3_cache_dir="/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong/cache/", # Local cache directory
        return_surya_stack=True,
        dst_hdf5_path=dst_data_path,
        delay_days=3,
        max_number_of_samples=None,
        storm_threshold=-50 # Validate on moderate storms
    )
    
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    # 3. Load Model Architecture
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
        dropout=config["model"]["dropout"],
        num_penultimate_transformer_layers=0,
        num_penultimate_heads=0,
        num_outputs=216,
        config=config,
    )

    # 4. Load Weights from Checkpoint
    print(f"📥 Loading weights from {CKPT_PATH}...")
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    
    # Handle "state_dict" key if present (Lightning checkpoints have it)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
    # Fix LoRA/Prefix keys if needed (remove 'model.' prefix if saved that way)
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    
    # Load with strict=False to ignore LoRA wrapper mismatches if necessary
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.cuda()

    # 5. Run Inference
    all_preds = []
    all_targets = []
    
    print("🔮 Running inference on validation set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", unit="batch"):
        # for batch in val_loader:
            # Move to GPU
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            
            # Forward Pass
            preds = model(batch) # Output: [Batch, 216]
            
            # Get Targets (Dst)
            targets = batch["forecast"] # Output: [Batch, 3, 216]
            if targets.dim() == 3:
                targets = targets[:, 0, :] # Take first step if multiple provided
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0).flatten()
    y_true = np.concatenate(all_targets, axis=0).flatten()

    return y_true, y_pred

# ==========================================
# PART 3: Generate Figures
# ==========================================
def plot_results(y_true, y_pred):
    print("📈 Generating Evaluation Plots...")

    # Figure 1: Time Series Comparison (First 200 points for clarity)
    plt.figure(figsize=(12, 6))
    limit = 200 
    plt.plot(y_true[:limit], label="Ground Truth (Dst)", color='black', linewidth=1.5)
    plt.plot(y_pred[:limit], label="Prediction", color='red', linestyle='--', linewidth=1.5)
    plt.title(f"Forecast vs Actual (First {limit} validation samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Dst Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{FIG_DIR}/time_series_comparison.png", dpi=300)
    print(f"   -> Saved to {FIG_DIR}/time_series_comparison.png")

    # Figure 2: Scatter Plot (Correlation)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10, color='blue')
    
    # Perfect fit line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Perfect Fit")
    
    plt.title("Scatter Plot: Ground Truth vs Prediction")
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig(f"{FIG_DIR}/scatter_plot.png", dpi=300)
    print(f"   -> Saved to {FIG_DIR}/scatter_plot.png")

if __name__ == "__main__":
    # 1. Plot Loss History
    plot_loss_curves()
    
    # 2. Run Inference
    y_true, y_pred = run_evaluation()
    
    # 3. Plot Predictions
    plot_results(y_true, y_pred)
    
    print("\n✅ Visualization Complete! Check the 'Figs' folder.")