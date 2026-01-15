from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SURYA_ROOT = PROJECT_ROOT / "Surya"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURYA_ROOT))

import os
import yaml
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
import torch
from torch.utils.data import DataLoader

# import torch.multiprocessing as mp
# mp.set_sharing_strategy("file_system")
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from downstream_apps.template_jinsu.datasets.dataset_flare import SolarFlareDataset
from downstream_apps.template_jinsu.lightning_modules.headmodule import FlareDSModel
from surya.models.helio_spectformers import HelioSpectFormer
from surya.utils.data import build_scalers  # Data scaling utilities for Surya stacks
from workshop_infrastructure.utils import apply_peft_lora


def train(config):

    scalers = build_scalers(info=config["data"]["scalers"])

    # Dataset and Dataloader
    train_dataset = SolarFlareDataset(
        #### All these lines are required by the parent HelioNetCDFDataset class
        index_path=config["data"]["train_data_path"],
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        n_input_timestamps=config["model"]["time_embedding"]["time_dim"],
        rollout_steps=config["rollout_steps"],
        channels=config["data"]["channels"],
        drop_hmi_probability=config["drop_hmi_probablity"],
        use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
        scalers=scalers,
        phase="train",
        s3_use_simplecache=False,
        s3_cache_dir="/tmp/helio_s3_cache",
        #### Put your donwnstream (DS) specific parameters below this line
        label_type=config["data"]["downstream"]["label_type"],
        return_surya_stack=True,
        max_number_of_samples=None,
        flare_index_path=config["data"]["downstream"]["train_index_path"],
    )

    # The Validation dataset changes the index we read
    val_dataset = SolarFlareDataset(
        #### All these lines are required by the parent HelioNetCDFDataset class
        index_path=config["data"]["valid_data_path"],
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        n_input_timestamps=config["model"]["time_embedding"]["time_dim"],
        rollout_steps=config["rollout_steps"],
        channels=config["data"]["channels"],
        drop_hmi_probability=config["drop_hmi_probablity"],
        use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
        scalers=scalers,
        phase="validation",
        s3_use_simplecache=False,
        s3_cache_dir="/tmp/helio_s3_cache",
        #### Put your donwnstream (DS) specific parameters below this line
        label_type=config["data"]["downstream"]["label_type"],
        return_surya_stack=True,
        max_number_of_samples=None,
        flare_index_path=config["data"]["downstream"]["val_index_path"],
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_data_workers"],
        multiprocessing_context="spawn",
        persistent_workers=True,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_data_workers"],
        multiprocessing_context="spawn",
        persistent_workers=True,
        pin_memory=True,
    )

    # set random seed
    L.seed_everything(config["seed_num"], workers=True)

    # define backbone
    backbone = HelioSpectFormer(
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
    )

    # load pretrained Surya weights
    model_state = backbone.state_dict()
    checkpoint_state = torch.load(
        config["pretrained_path"], weights_only=True, map_location="cpu"
    )
    filtered_checkpoint_state = {
        k: v
        for k, v in checkpoint_state.items()
        if k in model_state and v.shape == model_state[k].shape
    }

    model_state.update(filtered_checkpoint_state)
    backbone.load_state_dict(model_state, strict=True)

    if config["model"]["use_lora"]:
        backbone = apply_peft_lora(backbone, config)
    else:
        for name, param in backbone.named_parameters():
            if "embedding" in name or "backbone" in name:
                param.requires_grad = False
        parameters_with_grads = []
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                parameters_with_grads.append(name)
        print(
            f"{len(parameters_with_grads)} parameters require gradients: {', '.join(parameters_with_grads)}."
        )

    model = FlareDSModel(
        backbone=backbone,
        optimizer_dict=config["opt"],
        scheduler_dict=config["scheduler"],
        eval_threshold=config["downstream_model"]["threshold"],
        hidden_channels=config["downstream_model"]["hidden_channels"],
        dropout=config["downstream_model"]["dropout"],
    )

    # define wandb
    wandb_logger = WandbLogger(
        entity="surya_handson",
        project=config["wandb"]["project_name"],
        name=config["wandb"]["run_name"],
        log_model=False,
        save_dir=config["wandb"]["save_dir"],
    )

    csv_logger = CSVLogger("runs", name="simple_flare")

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            save_last=True,
            verbose=True,
            enable_version_counter=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        max_epochs=config["opt"]["epoch"],
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        precision=config["trainer"]["precision"],
        logger=[wandb_logger, csv_logger],
        callbacks=callbacks,
        log_every_n_steps=config["trainer"]["log_every_n_steps"],
        limit_train_batches=config["trainer"]["limit_train_batches"],
        limit_val_batches=config["trainer"]["limit_val_batches"],
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=(
            Path(config["downstream_model"]["ckpt_path"])
            / config["downstream_model"]["ckpt_file"]
            if config["downstream_model"]["resume"]
            else None
        ),
    )


if __name__ == "__main__":

    # Configuration paths - modify these if your files are in different locations
    parser = argparse.ArgumentParser(description="Train flare prediction model")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config_flare.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()
    config_path = args.config

    # Load configuration
    print("Loading configuration...")
    try:
        config = yaml.safe_load(open(config_path, "r"))
        config["data"]["scalers"] = yaml.safe_load(
            open(config["data"]["scalers_path"], "r")
        )
        print("Configuration loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure config.yaml exists in your current directory")
        raise

    train(config)
