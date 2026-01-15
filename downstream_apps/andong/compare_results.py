import os
import sys
import yaml
import torch
import inspect
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from datetime import timedelta

# --- 1. SETUP PATHS & IMPORTS ---
# Adjust these relative paths to match your folder structure
sys.path.append("../../")
sys.path.append("../../Surya")

from surya.utils.data import build_scalers
from workshop_infrastructure.utils import apply_peft_lora
from workshop_infrastructure.models.finetune_models import HelioSpectformer1D
from downstream_apps.andong.datasets.dataset_andong import DstDataset

# Avoid CUDA OOM / Deadlocks
torch.set_float32_matmul_precision('medium')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- 2. ROBUST MODEL LOADER (Fixes TypeErrors) ---
def load_model_robust(config, checkpoint_path=None, is_lora=False):
    """
    Initializes the model by filtering config arguments that don't match 
    the class signature, preventing 'unexpected keyword' errors.
    """
    model_args = config["model"].copy()
    
    # Map common naming mismatches
    if "in_channels" in model_args:
        model_args["in_chans"] = model_args.pop("in_channels")
    if "spectral_blocks" in model_args:
        model_args["n_spectral_blocks"] = model_args.pop("spectral_blocks")

    # Dynamic Filtering: Only pass args that exist in __init__
    sig = inspect.signature(HelioSpectformer1D.__init__)
    valid_keys = set(sig.parameters.keys())
    filtered_args = {k: v for k, v in model_args.items() if k in valid_keys}
    
    # Initialize Model
    # Note: num_outputs=216 matches your notebook configuration
    model = HelioSpectformer1D(num_outputs=216, config=config, **filtered_args)

    # Apply LoRA Structure (if needed)
    if is_lora:
        model = apply_peft_lora(model, config)

    # Load Weights
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        
        # Clean 'model.' prefix from Lightning
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        
        # Shape Filtering (Skip layers with mismatched shapes, e.g. input layer)
        model_state = model.state_dict()
        filtered_state = {
            k: v for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        
        # Strict=False allows us to skip the mismatched input layer (26 vs 13 channels)
        model.load_state_dict(filtered_state, strict=False)
    else:
        print(f"⚠️ Warning: Checkpoint not found at {checkpoint_path}")

    return model.eval().cuda()

def main():
    # --- 3. CONFIGURATION ---
    config_path = "./configs/config.yaml"
    config = yaml.safe_load(open(config_path, "r"))
    config["data"]["scalers"] = yaml.safe_load(open(config["data"]["scalers_path"], "r"))
    scalers = build_scalers(info=config["data"]["scalers"])

    # --- 4. DATASET (Validation Only) ---
    print("Initializing Dataset...")
    dataset = DstDataset(
        index_path=config["data"]["valid_data_path"],
        # Pulling params from your config structure
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        n_input_timestamps=config["model"]["time_embedding"]["time_dim"],
        rollout_steps=config["rollout_steps"],
        channels=config["data"]["channels"],
        drop_hmi_probability=config["drop_hmi_probability"],
        use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
        scalers=scalers,
        phase="val",
        # Local paths from your notebook
        dst_hdf5_path="/media/faraday/andong/Dataspace/GONG_NN/Data/ML_Ready_Dataset_2019-2026-3h.h5",
        delay_days=3,
        return_surya_stack=True,
        # No storm threshold for validation to see general performance
        storm_threshold=None 
    )

    # Use num_workers=0 to avoid the "Deadlock" mentioned in your screenshot
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # --- 5. MODEL SETUP ---
    print("Loading Baseline Model...")
    # Baseline uses ONLY the pre-trained weights (Zero-Shot)
    baseline_model = load_model_robust(config, config["pretrained_path"], is_lora=False)

    print("Loading Fine-Tuned Model...")
    # Update this path to your best checkpoint from the 'runs/' folder
    # Example: "runs/dst_forecast/version_0/checkpoints/epoch=9-step=1000.ckpt"
    # For now, we reuse pretrained path as a placeholder if you don't have a specific checkpoint yet
    ft_checkpoint = "runs/dst_forecast/dst_finetune_3day_delay/checkpoints/last.ckpt" 
    
    # Check if fine-tuned checkpoint exists, otherwise warn user
    if not os.path.exists(ft_checkpoint):
        print(f"⚠️ Fine-tuned checkpoint not found at {ft_checkpoint}. Plot will show 2 identical lines.")
        ft_checkpoint = config["pretrained_path"]
        
    ft_model = load_model_robust(config, ft_checkpoint, is_lora=True)

    # --- 6. INFERENCE & PLOTTING ---
    actuals, base_preds, ft_preds = [], [], []
    
    print("Running Inference (First 100 samples)...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 100: break # Limit samples for speed
            
            imgs = batch["images"].cuda()
            target = batch["forecast"] # Shape [1, 1, 216] or [1, 216]

            # Get Scalar Values (assuming index 0 is the immediate forecast)
            # Target might be [Batch, Rollout] -> take first step
            if target.dim() == 3: val_target = target[0, 0, 0].item()
            else: val_target = target[0, 0].item()
            
            p_base = baseline_model(imgs) # [1, 216]
            p_ft = ft_model(imgs)         # [1, 216]

            actuals.append(val_target)
            base_preds.append(p_base[0, 0].item())
            ft_preds.append(p_ft[0, 0].item())

    # Generate Plot
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label="Real Dst (Ground Truth)", color="black", linewidth=2)
    plt.plot(base_preds, label="Baseline (Zero-Shot)", color="gray", linestyle="--")
    plt.plot(ft_preds, label="Fine-Tuned (LoRA)", color="red", linewidth=2)
    
    plt.title(f"3-Day Dst Forecast Comparison (First {len(actuals)} Valid Samples)")
    plt.xlabel("Sample Index (3h steps)")
    plt.ylabel("Dst (nT)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = "comparison_results.png"
    plt.savefig(out_file)
    print(f"✅ Comparison saved to {out_file}")

if __name__ == "__main__":
    main()