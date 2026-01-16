import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# Add paths to ensure imports work
sys.path.append("../")
sys.path.append("../Surya")

from wsa_surya_training.models.wsa_model_head import WSAModel
from wsa_surya_training.lightning_modules.wsa_lightning_module import WSALightningModule
from wsa_surya_training.datasets.wsa_dataset import WSAImageDataset
from surya.utils.data import build_scalers
from workshop_infrastructure.models.finetune_models import HelioSpectformer2D
# from surya.utils.data import build_scalers

# --- CONFIGURATION ---
CHECKPOINT_PATH = "./checkpoints/wsa_best_val_loss=0.1136.ckpt"
CONFIG_PATH = "./configs/wsa_config.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_from_checkpoint(ckpt_path, config):
    print(f"Loading model from {ckpt_path}...")
    
    # 1. Dummy config
    dummy_model_config = {
        "model": {
            "ft_unembedding_type": "linear", 
            "ft_head_type": "linear",
            "ft_out_chans": 13,
            "ft_layers": 1
        }
    }

    # 2. Re-initialize the Encoder 
    encoder = HelioSpectformer2D(
        img_size=4096,
        patch_size=16,
        in_chans=13,
        embed_dim=1280,
        time_embedding={'type': 'linear', 'time_dim': 1},
        depth=10,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.1,
        window_size=2,
        dp_rank=1,
        n_spectral_blocks=1,
        config=dummy_model_config
    )

    # --- THE FIX: Correctly utilize the SpectFormer Backbone ---
    class ManualForwardEncoderWrapper(nn.Module):
        def __init__(self, base_encoder):
            super().__init__()
            self.base_encoder = base_encoder
            
            # Verify backbone exists (based on your dump)
            if not hasattr(base_encoder, 'backbone'):
                raise AttributeError("Expected 'backbone' in encoder based on architecture dump.")

        def forward(self, batch):
            # 1. Run Embedding (Level 1)
            # This handles the [Batch, C, H, W] -> [Batch, SeqLen, Dim] conversion
            if isinstance(batch, dict):
                x_input = batch['ts']
                dt_input = batch.get('time_delta_input', 0.0)
                if isinstance(dt_input, float):
                    dt_input = torch.tensor(dt_input, device=x_input.device, dtype=x_input.dtype)
                
                # Call Level 1 embedding
                x = self.base_encoder.embedding(x_input, dt_input)
            else:
                dt_dummy = torch.tensor(0.0, device=batch.device, dtype=batch.dtype)
                x = self.base_encoder.embedding(batch, dt_dummy)
            
            # 2. Run Backbone (SpectFormer)
            # Your dump showed 'backbone' contains the blocks (spectral + attention).
            # Calling it directly runs those blocks and returns tokens.
            x = self.base_encoder.backbone(x)
            
            # 3. Handle Reshaping (Tokens -> Grid)
            if x.ndim == 4: # [B, T, N, C]
                B, T, N, C = x.shape
                x = x.reshape(B * T, N, C)
                
            B, N, C = x.shape
            H_grid = W_grid = int(N**0.5) 
            
            if not hasattr(self, "printed"):
                print(f"   [Debug] Token Shape: {x.shape} -> Reshaping to [{B}, {C}, {H_grid}, {W_grid}]")
                self.printed = True
                
            x = x.transpose(1, 2)
            x = x.reshape(B, C, H_grid, W_grid)
            return x

    # Wrap the encoder
    safe_encoder = ManualForwardEncoderWrapper(encoder)

    # 3. Re-initialize the WSA Model Wrapper
    wsa_model = WSAModel(
        encoder=safe_encoder, 
        encoder_out_channels=config["model"]["embed_dim"],
        freeze_encoder=False 
    )

    # 4. Initialize Lightning Module
    dummy_metrics = {
        "train_loss": lambda x, y: ({}, []), 
        "train_metrics": lambda x, y: ({}, []), 
        "val_metrics": lambda x, y: ({}, [])
    }
    
    lit_model = WSALightningModule(
        model=wsa_model,
        metrics=dummy_metrics,
        lr=0.0
    )
    
    # 5. Manual Weight Loading
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    missing, unexpected = lit_model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    print(f"Weights loaded. Ignored {len(missing)} missing keys.")
    
    lit_model.eval()
    lit_model.to(DEVICE)
    return lit_model

def plot_comparison(input_tensor, target, prediction, save_path=None):
    # Move to CPU and numpy
    # input: [1, 13, T, H, W]. Take 1st channel.
    aia_img = input_tensor[0, 0, 0].cpu().numpy() 
    
    target_img = target.squeeze().cpu().numpy()
    pred_img = prediction.squeeze().cpu().numpy()

    # Mask off-disk pixels (0.0)
    mask = target_img == 0
    target_img = np.where(mask, np.nan, target_img)
    pred_img = np.where(mask, np.nan, pred_img)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    try:
        cmap_aia = 'sdoaia193' 
        plt.get_cmap(cmap_aia)
    except:
        cmap_aia = 'hot'

    im1 = axes[0].imshow(aia_img, cmap=cmap_aia, origin='lower', vmin=0, vmax=1)
    axes[0].set_title("Input: AIA 193Å")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(target_img, cmap='plasma', origin='lower', vmin=250, vmax=700)
    axes[1].set_title("Ground Truth: WSA Speed (km/s)")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    im3 = axes[2].imshow(pred_img, cmap='plasma', origin='lower', vmin=250, vmax=700)
    axes[2].set_title("Prediction: WSA Speed (km/s)")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.show()

def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    scalers_path = config["data"]["scalers_path"]
    with open(scalers_path, "r") as f:
        scalers_config = yaml.safe_load(f)
    scalers = build_scalers(info=scalers_config)

    model = load_model_from_checkpoint(CHECKPOINT_PATH, config)

    val_dataset = WSAImageDataset(
        cr_list=config["data"]["val_crs"],
        phase="val",
        surya_index_path=config["data"]["surya_index_path"],
        wsa_map_dir=config["data"]["wsa_map_dir"],
        wsa_params_path=config["data"]["wsa_params_path"],
        scalers=scalers,
        channels=config["data"]["channels"],
        normalize_wsa=True,
        tolerance_days=config["data"]["cr_tolerance_days"],
        s3_use_simplecache=config["data"]["s3_use_simplecache"],
        s3_cache_dir=config["data"]["s3_cache_dir"],
    )

    print(f"Validation set size: {len(val_dataset)}")
    
    def to_tensor(data):
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            return data.float()
        return data

    indices_to_check = [0, 1] 
    
    for idx in indices_to_check:
        if idx >= len(val_dataset): continue
        
        sample = val_dataset[idx]
        
        # Prepare inputs
        x_raw = to_tensor(sample["ts"]).unsqueeze(0).to(DEVICE)
        target = to_tensor(sample["wsa_map"]).unsqueeze(0).to(DEVICE)
        
        # Expand to 13 channels
        x_expanded = x_raw.expand(-1, 13, -1, -1, -1)
        
        print(f"Plotting index {idx}...")
        
        with torch.no_grad():
            pred = model(x_expanded)
        
        plot_comparison(
            input_tensor=x_expanded, 
            target=target, 
            prediction=pred, 
            save_path=f"result_plot_{idx}.png"
        )

if __name__ == "__main__":
    main()