import os
import logging
import sys
from typing import Dict

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model

from packaging.version import Version
import wandb

if Version(wandb.__version__) < Version("0.20.0"):
    _WANDB_USE_SYNC = True
else:
    _WANDB_USE_SYNC = False


# ---------------------------------------------------------------------------
# Distributed utilities
# ---------------------------------------------------------------------------

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------

def create_logger(output_dir: str, dist_rank: int, name: str) -> logging.Logger:
    """Create a file+console logger identified by name and rank."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = "[%(asctime)s %(name)s]: %(levelname)s %(message)s"

    if name.endswith("main"):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(output_dir, f"{name}.log"), mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    return logger


def log(run, data: dict, step=None, commit=None) -> None:
    """Thin wrapper around wandb.run.log that handles API version differences."""
    if run is not None:
        if _WANDB_USE_SYNC:
            run.log(data, step, commit)
        else:
            run.log(data, step, commit)
    else:
        print(data)


# ---------------------------------------------------------------------------
# Scaler utilities
# ---------------------------------------------------------------------------

def class_from_name(module_name: str, class_name: str):
    """Import and return a class by its module and class name strings."""
    m = __import__(module_name, globals(), locals(), [class_name])
    return getattr(m, class_name)


def build_scalers(info) -> Dict:
    """Reconstruct per-channel scaler objects from a scalers YAML file or dict.

    Args:
        info: Path to a scalers YAML file, or an already-loaded dict.

    The YAML entries contain a 'base' module path and 'class' name that were
    recorded when the scalers were originally fitted (e.g. 'surya.datasets.transformations').
    We resolve the class from our local transformations module instead of the
    stored (now stale) module path.
    """
    import yaml
    import workshop_infrastructure.datasets.transformations as _transformations

    if not isinstance(info, dict):
        with open(info, "r", encoding="utf-8") as f:
            info = yaml.safe_load(f)

    ret_dict = {k: None for k in info.keys()}
    for p_key, p_val in info.items():
        cls = getattr(_transformations, p_val["class"])
        ret_dict[p_key] = cls.from_dict(p_val)
    return ret_dict

def apply_peft_lora(
    model: torch.nn.Module,
    lora_config,
) -> torch.nn.Module:
    """
    Applies PEFT LoRA adapters to a model.

    Args:
        model: The model to apply LoRA to.
        lora_config: A LoraAdapterConfig dataclass or a plain dict with the
                     same fields (r, lora_alpha, target_modules, lora_dropout, bias).

    Returns:
        Model with PEFT LoRA adapters applied.
    """
    # Accept both a dataclass and a plain dict for flexibility.
    if hasattr(lora_config, "__dataclass_fields__"):
        cfg = lora_config
        r, lora_alpha = cfg.r, cfg.lora_alpha
        target_modules, lora_dropout, bias = cfg.target_modules, cfg.lora_dropout, cfg.bias
    else:
        r = lora_config.get("r", 8)
        lora_alpha = lora_config.get("lora_alpha", 8)
        target_modules = lora_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"])
        lora_dropout = lora_config.get("lora_dropout", 0.1)
        bias = lora_config.get("bias", "none")

    print(f"Applying PEFT LoRA: r={r}, alpha={lora_alpha}, dropout={lora_dropout}, modules={target_modules}")

    # Create LoRA configuration
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)

    # Log the number of trainable parameters
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )

    return model