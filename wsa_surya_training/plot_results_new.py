import torch
import yaml
from pathlib import Path
from pathlib import Path
import sys
import os

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))  # Add this line
SURYA_DIR = PROJECT_ROOT / "Surya"
sys.path.insert(0, str(SURYA_DIR))

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CHECKPOINT_PATH = Path("/home/arpitkumar/surya_workshop/wsa_surya_training/checkpoints/wsa_best_val_loss=0.1136.ckpt")
from surya.utils.data import build_scalers


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path("/home/arpitkumar/surya_workshop/wsa_surya_training/checkpoints/wsa_best_val_loss=0.1136.ckpt")
CONFIG_PATH = SCRIPT_DIR / "configs/wsa_config.yaml"

def load_yaml(path):
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint

def to_tensor(x):
    """Convert numpy array to tensor."""
    if isinstance(x, torch.Tensor):
        return x
    import numpy as np
    return torch.from_numpy(np.array(x)).float()

def plot_comparison(input_tensor, target, prediction, save_path):
    """Plot input, target, and prediction side by side."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input (use first channel)
    # input_np = input_tensor[0, 0].cpu().numpy()
    input_np = input_tensor[0, 0].cpu().numpy().squeeze()
    if input_np.ndim == 3: input_np = input_np[0]
    axes[0].imshow(input_np, cmap='viridis')
    axes[0].set_title('Input (AIA 193)')
    axes[0].axis('off')
    
    # Target
    # target_np = target[0, 0].cpu().numpy()
    target_np = target[0, 0].cpu().numpy().squeeze()
    axes[1].imshow(target_np, cmap='viridis')
    axes[1].set_title('Target (WSA Map)')
    axes[1].axis('off')
    
    # Prediction
    # pred_np = prediction[0, 0].cpu().detach().cpu().numpy()
    pred_np = prediction[0, 0].detach().cpu().numpy().squeeze()
    print(f"Prediction range: {np.min(pred_np):.4f} to {np.max(pred_np):.4f}")
    vmin = np.percentile(pred_np, 1)
    vmax = np.percentile(pred_np, 99)
    axes[2].imshow(pred_np, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2].set_title(f'Prediction\nRange: [{vmin:.2f}, {vmax:.2f}]')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot to {save_path}")
    plt.close()

def main():
    """Main plotting function."""
    print(f"📊 Loading checkpoint from {CHECKPOINT_PATH}...")
    
    if not CHECKPOINT_PATH.exists():
        print(f"❌ Checkpoint file not found: {CHECKPOINT_PATH}")
        return
    
    # Load configuration
    print(f"📋 Loading configuration from {CONFIG_PATH}...")
    config = load_yaml(CONFIG_PATH)
    
    # Load scalers configuration if it is a path string
    if isinstance(config["data"]["scalers_path"], str):
        config["data"]["scalers_path"] = load_yaml(config["data"]["scalers_path"])

    # Load checkpoint
    checkpoint = load_checkpoint(CHECKPOINT_PATH, DEVICE)
    
    # Initialize model
    print("🏗️ Initializing model...")
    from workshop_infrastructure.models.finetune_models import HelioSpectformer2D
    
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
        n_spectral_blocks=config["model"]["n_spectral_blocks"],
        config=config,
        finetune=True,
    )
    
    # Load model weights from checkpoint
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully!")
    
    # Load validation dataset
    print("📚 Loading validation dataset...")
    from wsa_surya_training.datasets.wsa_dataset import WSAImageDataset
    
    # val_dataset = WSAImageDataset(
    #     data_dir=config["data"]["data_dir"],
    #     split="val",
    #     scalers_path=config["data"]["scalers_path"]
    # )
        # Common parameters
    scalers = build_scalers(info=config["data"]["scalers_path"])
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
    
        # Create validation dataset
    val_dataset = WSAImageDataset(
        cr_list=config["data"]["val_crs"],
        phase="val",
        **dataset_kwargs
    )
    
    print(f"✅ Loaded {len(val_dataset)} validation samples")
    
    # Generate plots for random samples
    import random
    num_plots = min(5, len(val_dataset))
    indices = random.sample(range(len(val_dataset)), num_plots)
    
    print(f"\n🎨 Generating {num_plots} result plots...")
    
    for idx in indices:
        try:
            sample = val_dataset[idx]
            
            # Prepare inputs
            x_raw = to_tensor(sample["ts"]).unsqueeze(0).to(DEVICE)
            target = to_tensor(sample["wsa_map"]).unsqueeze(0).to(DEVICE)
            
            # Expand to 13 channels if needed
            if x_raw.shape[1] == 1:
                x_expanded = x_raw.expand(-1, 13, -1, -1, -1)
            else:
                x_expanded = x_raw
            
            

            # Prepare batch dictionary for the model
            batch_input = {
                "ts": x_expanded,
                #"time_delta_input": to_tensor(sample.get("time_delta_input", 0.0)).to(DEVICE),
                "time_delta_input": to_tensor(sample.get("time_delta_input", 0.0)).unsqueeze(0).to(DEVICE)
            }
            
            print(f"Plotting index {idx}...")
            
            with torch.no_grad():
                #pred = model(x_expanded)
                pred = model(batch_input)
            
            plot_comparison(
                input_tensor=x_expanded, 
                target=target, 
                prediction=pred, 
                save_path=f"result_plot_{idx}.png"
            )
        except Exception as e:
            print(f"❌ Error plotting index {idx}: {e}")
            continue
    
    print("\n✅ Plotting complete!")

if __name__ == "__main__":
    main()