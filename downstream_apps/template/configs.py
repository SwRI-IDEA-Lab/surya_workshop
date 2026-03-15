"""
Typed configuration dataclasses for the flare-forecasting template app.

load_config() is the single entry point: it reads config_script.yaml and
returns a fully typed TrainingConfig. All downstream code receives a
TrainingConfig instead of a raw dict, giving IDE completion and
catching missing/misspelled keys at startup rather than mid-training.

The YAML is divided into five sections (data, model, training, output, logging).
load_config() maps each section to its dataclass explicitly, so the relationship
between YAML keys and Python fields is easy to follow.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields
from pathlib import Path
from typing import List, Optional, Union

import yaml

from workshop_infrastructure.configs import LoraAdapterConfig, ModelConfig, TimeEmbeddingConfig


def _from_dict(cls, d: dict):
    """Construct a dataclass from a dict, silently ignoring unknown keys.

    Prevents load_config() from needing to list every field explicitly,
    so adding a new field to a dataclass only requires editing the dataclass
    and YAML — not load_config() as well.
    """
    known = {f.name for f in dc_fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class DataConfig:
    """Paths, sampling parameters, and S3 cache settings for the SDO/flare dataset."""
    train_data_path: str
    valid_data_path: str
    scalers_path: str
    channels: List[str]
    time_delta_input_minutes: List[int]
    time_delta_target_minutes: int
    flare_index_path: str
    # Downstream label-alignment parameters (passed to FlareDSDataset)
    ds_time_column: str = "start_time"
    ds_time_tolerance: str = "4d"
    ds_match_direction: str = "forward"
    # S3
    s3_anon: bool = False
    s3_cache_dir: Optional[str] = None
    # Development
    max_samples: Optional[int] = None


@dataclass
class OutputConfig:
    """Paths and S3 settings for checkpoints and artifacts."""
    ckpt_dir: str = "checkpoints"
    # S3 upload of best checkpoint (all three required together; leave s3_bucket null to disable)
    s3_bucket: Optional[str] = None
    s3_prefix: str = ""
    s3_best_key: str = "best.ckpt"


@dataclass
class TrainingConfig:
    """Top-level configuration for a fine-tuning run."""
    job_id: str
    data: DataConfig
    model: ModelConfig
    output: OutputConfig = field(default_factory=OutputConfig)
    learning_rate: float = 1e-4
    max_epochs: int = 20
    batch_size: int = 2
    rollout_steps: int = 0
    drop_hmi_probability: float = 0.0
    use_latitude_in_learned_flow: bool = False
    dtype: str = "float32"
    wandb_project: str = "template_flare_regression"
    wandb_entity: Optional[str] = None


def load_config(path: Union[str, Path]) -> TrainingConfig:
    """Parse config_script.yaml into a typed TrainingConfig.

    The YAML has five top-level sections (data, model, training, output, logging)
    plus job_id. Each section is parsed into its corresponding dataclass.
    Unknown keys within each section are silently ignored by _from_dict(), so
    adding a new field only requires editing the dataclass and YAML.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # model: build nested configs first, then construct ModelConfig.
    # pretrained_path lives in the model: YAML section and on ModelConfig — not lifted to TrainingConfig.
    model_raw = raw["model"]
    model_kwargs = dict(model_raw)
    model_kwargs["time_embedding"] = _from_dict(TimeEmbeddingConfig, model_raw.get("time_embedding", {}))
    model_kwargs["lora_config"] = _from_dict(LoraAdapterConfig, model_raw.get("lora_config", {}))
    model_cfg = _from_dict(ModelConfig, model_kwargs)

    training = raw.get("training", {})
    logging_cfg = raw.get("logging", {})

    return TrainingConfig(
        job_id=raw["job_id"],
        data=_from_dict(DataConfig, raw["data"]),
        model=model_cfg,
        output=_from_dict(OutputConfig, raw.get("output", {})),
        learning_rate=training.get("learning_rate", 1e-4),
        max_epochs=training.get("max_epochs", 20),
        batch_size=training.get("batch_size", 2),
        rollout_steps=training.get("rollout_steps", 0),
        drop_hmi_probability=training.get("drop_hmi_probability", 0.0),
        use_latitude_in_learned_flow=training.get("use_latitude_in_learned_flow", False),
        dtype=training.get("dtype", "float32"),
        wandb_project=logging_cfg.get("wandb_project", "template_flare_regression"),
        wandb_entity=logging_cfg.get("wandb_entity"),
    )
