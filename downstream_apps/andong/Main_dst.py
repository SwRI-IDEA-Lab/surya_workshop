import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
import wandb

# --- SETUP PATHS (Adjust these if your folder structure changes) ---
# Add paths to Workshop infrastructure and Surya source code
sys.path.append("../../")
sys.path.append("../../Surya")

from surya.utils.data import build_scalers
from workshop_infrastructure.utils import apply_peft_lora
from workshop_infrastructure.models.finetune_models import HelioSpectformer1D
from downstream_apps.template.lightning_modules.pl_simple_baseline import FlareLightningModule
from downstream_apps.andong.datasets.dataset_andong import DstDataset  # Ensure this import path matches your file structure

# --- 1. CONFIGURATION (No AWS Credentials Needed) ---
def setup_environment():
    # 1. Memory Optimizations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.set_float32_matmul_precision('medium')
    
    # 2. AWS Region (REQUIRED even for public/anonymous data)
    # The data lives in Oregon (us-west-2). If you don't set this, it might default to east-1 and fail.
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2" 
    
    # Note: We removed AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.
    # The dataset class will now default to Anonymous mode (or look for ~/.aws/credentials if it exists).

    # 3. WandB Login
    if "WANDB_API_KEY" not in os.environ:
        # It's better to export this in your terminal, but hardcoding works for testing
        os.environ["WANDB_API_KEY"] = "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"

    # WandB Directories
    os.environ["WANDB_DIR"] = "./wandb/wandb_logs"
    os.environ["WANDB_CACHE_DIR"] = "./wandb/wandb_cache"
    os.environ["WANDB_CONFIG_DIR"] = "./wandb/wandb_config"
    
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
    os.makedirs(os.environ["WANDB_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["WANDB_CONFIG_DIR"], exist_ok=True)
    
    try:
        wandb.login()
    except:
        print("⚠️ WandB login failed. Logs will be local only.")

# --- 2. METRICS CLASS ---
class DstMetrics:
    def __init__(self, mode="train_loss"):
        self.mode = mode
        
    def __call__(self, preds, target):
        # 1. Fix Target Shape [Batch, 3, 216] -> [Batch, 216]
        if target.dim() == 3:
            target = target[:, 0, :]
            
        # 2. Fix Preds Shape [Batch, 216, 1] -> [Batch, 216]
        if preds.dim() == 3 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)
            
        # 3. Calculate Loss (MSE)
        loss = torch.nn.functional.mse_loss(preds, target)
        
        # 4. Return (Dict, Weights) ALWAYS
        if self.mode == "train_loss":
            return {"mse": loss}, [1]
        else:
            # FIX: Return a dummy second value to prevent crash during validation
            return {"mse": loss}, [] 

# --- 3. MAIN EXECUTION BLOCK ---
def main():
    setup_environment()
    
    # Global Seed
    L.seed_everything(42, workers=True)

    # Load Config
    config_path = "./configs/config.yaml"
    print(f"📋 Loading configuration from {config_path}...")
    
    try:
        config = yaml.safe_load(open(config_path, "r"))
        # Load Scalers
        config["data"]["scalers"] = yaml.safe_load(open(config["data"]["scalers_path"], "r"))
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    scalers = build_scalers(info=config["data"]["scalers"])

    # --- Dataset Setup ---
    # Path to your LOCAL copy of the HDF5 data
    dst_data_path = "/media/faraday/andong/Dataspace/GONG_NN/Data/ML_Ready_Dataset_2019-2026-3h.h5"
    STORM_LIMIT = -300.0 # Filter threshold

    print("Initializing Datasets...")
    
    # CACHE DIR: Use your workspace storage to prevent /tmp filling up
    CACHE_DIR = "/home/andonghu/workspace/surya_workshop/cache"
    os.makedirs(CACHE_DIR, exist_ok=True)

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
        s3_use_simplecache=True,
        s3_cache_dir=CACHE_DIR,
        
        # Dst Specifics
        return_surya_stack=True,
        dst_hdf5_path=dst_data_path,
        delay_days=3,
        max_number_of_samples=None,
        storm_threshold=STORM_LIMIT
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
        s3_use_simplecache=True,
        s3_cache_dir=CACHE_DIR,
        
        # Dst Specifics
        return_surya_stack=True,
        dst_hdf5_path=dst_data_path,
        delay_days=3,
        max_number_of_samples=None,
        storm_threshold=STORM_LIMIT # Filter validation too?
    )

    # --- DataLoaders ---
    batch_size = 1 # Keep small to fit in GPU memory
    
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2, # Set low (0-2) to avoid RAM explosion
        persistent_workers=False,
        pin_memory=True,
    )

    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=2,
        persistent_workers=False,
        pin_memory=True,
    )

    # --- Model Setup ---
    print("Initializing Model...")
    output_dim = 216 
    
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
        num_outputs=output_dim,
        config=config,
    )

    # --- Load Pretrained Weights ---
    print("Loading Pretrained Weights...")
    model_state = model.state_dict()
    checkpoint_state = torch.load(config["pretrained_path"], weights_only=True, map_location="cpu")
    
    filtered_checkpoint_state = {
        k: v for k, v in checkpoint_state.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    model_state.update(filtered_checkpoint_state)
    model.load_state_dict(model_state, strict=True)

    # --- Apply LoRA ---
    print("Applying LoRA...")
    model = apply_peft_lora(model, config)

    # --- Metrics ---
    metrics = {
        'train_loss': DstMetrics("train_loss"),
        'train_metrics': DstMetrics("train_metrics"),
        'val_metrics': DstMetrics("val_metrics")
    }

    # --- Lightning Module ---
    learning_rate = 1e-5
    lit_model = FlareLightningModule(model, metrics, lr=learning_rate, batch_size=batch_size)

    # --- Loggers ---
    project_name = "surya_dst_forecast"
    run_name = "dst_finetune_3day_delay_multiGPU"

    wandb_logger = WandbLogger(
        entity="surya_handson",
        project=project_name,
        name=run_name,
        log_model=False,
        save_dir=os.environ["WANDB_DIR"],
    )
    
    csv_logger = CSVLogger("runs", name="dst_forecast")

    # --- Trainer ---
    print("Starting Trainer...")
    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",      # Will use all GPUs allocated by torchrun
        strategy="ddp",      # Distributed Data Parallel for Multi-GPU
        precision="bf16-mixed", 
        logger=[wandb_logger, csv_logger],
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            )
        ],
        accumulate_grad_batches=8, # Virtual Batch Size = 1 * 8 = 8
        log_every_n_steps=2,
    )

    # --- Start Training ---
    trainer.fit(lit_model, train_data_loader, val_data_loader)

if __name__ == "__main__":
    main()