"""
Typed configuration dataclasses for the flare-forecasting template app.

load_config() is the single entry point: it reads config_script.yaml and
returns a fully typed TrainingConfig. All downstream code receives a
TrainingConfig instead of a raw dict, giving IDE completion and
catching missing/misspelled keys at startup rather than mid-training.

Only the fields actually used by this app are represented here. The
YAML may contain additional keys that are silently ignored — that is
intentional to keep the dataclass contract minimal and clear.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import yaml

from workshop_infrastructure.configs import LoraAdapterConfig, ModelConfig, TimeEmbeddingConfig


@dataclass
class DataConfig:
    """Paths and sampling parameters for the SDO/flare dataset."""
    train_data_path: str
    valid_data_path: str
    scalers_path: str
    channels: List[str]
    time_delta_input_minutes: List[int]
    time_delta_target_minutes: int
    flare_index_path: str


@dataclass
class TrainingConfig:
    """Top-level configuration for a fine-tuning run."""
    job_id: str
    data: DataConfig
    model: ModelConfig
    pretrained_path: Optional[str] = None
    learning_rate: float = 1e-4
    rollout_steps: int = 0
    drop_hmi_probability: float = 0.0
    use_latitude_in_learned_flow: bool = False
    dtype: str = "float32"
    wandb_project: str = "template_flare_regression"


def load_config(path: Union[str, Path]) -> TrainingConfig:
    """Parse config_script.yaml into a typed TrainingConfig.

    Nested dicts are converted to their respective dataclass types.
    Fields not present in the dataclasses are ignored, so legacy or
    unused YAML keys do not cause errors.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model_raw = raw["model"]

    time_emb = TimeEmbeddingConfig(**model_raw["time_embedding"])

    lora_raw = model_raw.get("lora_config", {})
    lora_cfg = LoraAdapterConfig(
        r=lora_raw.get("r", 8),
        lora_alpha=lora_raw.get("lora_alpha", 8),
        target_modules=lora_raw.get("target_modules", ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]),
        lora_dropout=lora_raw.get("lora_dropout", 0.1),
        bias=lora_raw.get("bias", "none"),
    )

    model_cfg = ModelConfig(
        img_size=model_raw["img_size"],
        patch_size=model_raw["patch_size"],
        in_channels=model_raw["in_channels"],
        embed_dim=model_raw["embed_dim"],
        depth=model_raw["depth"],
        spectral_blocks=model_raw["spectral_blocks"],
        num_heads=model_raw["num_heads"],
        mlp_ratio=model_raw["mlp_ratio"],
        drop_rate=model_raw["drop_rate"],
        window_size=model_raw["window_size"],
        dp_rank=model_raw["dp_rank"],
        rpe=model_raw["rpe"],
        learned_flow=model_raw["learned_flow"],
        init_weights=model_raw["init_weights"],
        checkpoint_layers=model_raw["checkpoint_layers"],
        ensemble=model_raw.get("ensemble"),
        finetune=model_raw["finetune"],
        nglo=model_raw["nglo"],
        time_embedding=time_emb,
        pooling=model_raw["pooling"],
        penultimate_linear_layer=model_raw.get("penultimate_linear_layer", True),
        dropout=model_raw.get("dropout", 0.2),
        freeze_backbone=model_raw.get("freeze_backbone", False),
        use_lora=model_raw.get("use_lora", True),
        lora_config=lora_cfg,
    )

    data_raw = raw["data"]
    data_cfg = DataConfig(
        train_data_path=data_raw["train_data_path"],
        valid_data_path=data_raw["valid_data_path"],
        scalers_path=data_raw["scalers_path"],
        channels=data_raw["channels"],
        time_delta_input_minutes=data_raw["time_delta_input_minutes"],
        time_delta_target_minutes=data_raw["time_delta_target_minutes"],
        flare_index_path=data_raw["flare_index_path"],
    )

    return TrainingConfig(
        job_id=raw["job_id"],
        data=data_cfg,
        model=model_cfg,
        pretrained_path=raw.get("pretrained_path"),
        learning_rate=raw.get("optimizer", {}).get("learning_rate", 1e-4),
        rollout_steps=raw.get("rollout_steps", 0),
        drop_hmi_probability=raw.get("drop_hmi_probability", 0.0),
        use_latitude_in_learned_flow=raw.get("use_latitude_in_learned_flow", False),
        dtype=raw.get("dtype", "float32"),
        wandb_project=raw.get("wandb_project", "template_flare_regression"),
    )
