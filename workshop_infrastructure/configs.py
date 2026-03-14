"""
Typed configuration dataclasses for the Surya backbone and LoRA adapter.

These dataclasses describe the model architecture and fine-tuning parameters
understood by HelioSpectformer1D / HelioSpectformer2D and apply_peft_lora().
They are intentionally kept separate from downstream task config (DataConfig,
TrainingConfig) so that new downstream apps can reuse ModelConfig without
pulling in task-specific fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TimeEmbeddingConfig:
    """Controls how temporal position is encoded before the transformer blocks."""
    type: str = "linear"         # "linear" | "fourier" | "perceiver"
    n_queries: Optional[int] = None  # Required for "perceiver"; unused otherwise
    time_dim: int = 1            # Number of input timesteps


@dataclass
class LoraAdapterConfig:
    """
    Settings for PEFT LoRA fine-tuning.

    Passed to apply_peft_lora(); mirrors the fields of peft.LoraConfig.
    Named LoraAdapterConfig to avoid confusion with peft.LoraConfig.
    """
    r: int = 8
    lora_alpha: int = 8
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    )
    lora_dropout: float = 0.1
    bias: str = "none"


@dataclass
class ModelConfig:
    """
    Full configuration for the HelioSpectFormer backbone and fine-tuning head.

    Backbone parameters mirror the HelioSpectFormer constructor.
    Fine-tuning head parameters are used by HelioSpectformer1D.
    """
    # --- Backbone ---
    img_size: int = 4096
    patch_size: int = 16
    in_channels: int = 13
    embed_dim: int = 1280
    depth: int = 10
    spectral_blocks: int = 2
    num_heads: int = 16
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    window_size: int = 2
    dp_rank: int = 4
    rpe: bool = False
    learned_flow: bool = False
    init_weights: bool = False
    checkpoint_layers: List[int] = field(default_factory=lambda: list(range(10)))
    ensemble: Optional[int] = None
    nglo: int = 1
    time_embedding: TimeEmbeddingConfig = field(default_factory=TimeEmbeddingConfig)

    # --- Fine-tuning head ---
    # One of: "global_average" | "global_max" | "attention" | "transformer" | "class_token"
    pooling: str = "class_token"
    penultimate_linear_layer: bool = True
    dropout: float = 0.2
    freeze_backbone: bool = False

    # --- LoRA ---
    use_lora: bool = True
    lora_config: LoraAdapterConfig = field(default_factory=LoraAdapterConfig)
