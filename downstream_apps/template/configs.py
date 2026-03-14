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

from dataclasses import dataclass, fields as dc_fields
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
    """Paths and sampling parameters for the SDO/flare dataset."""
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

    Nested dicts are converted to their dataclass types via _from_dict(),
    which silently drops YAML keys not present in the dataclass. This means
    adding a new field only requires editing the dataclass and YAML — not
    this function.

    The one exception is learning_rate, which lives under optimizer.learning_rate
    in the YAML but is a flat field on TrainingConfig.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model_raw = raw["model"]
    model_kwargs = dict(model_raw)
    model_kwargs["time_embedding"] = _from_dict(TimeEmbeddingConfig, model_raw.get("time_embedding", {}))
    model_kwargs["lora_config"] = _from_dict(LoraAdapterConfig, model_raw.get("lora_config", {}))
    model_cfg = _from_dict(ModelConfig, model_kwargs)

    training_kwargs = dict(raw)
    training_kwargs["learning_rate"] = raw.get("optimizer", {}).get("learning_rate", 1e-4)
    training_kwargs["data"] = _from_dict(DataConfig, raw["data"])
    training_kwargs["model"] = model_cfg
    return _from_dict(TrainingConfig, training_kwargs)
