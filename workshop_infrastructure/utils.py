import functools
import os
import logging
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from workshop_infrastructure.configs import LoraAdapterConfig


# ---------------------------------------------------------------------------
# AWS / infrastructure utilities
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def detect_ec2_region() -> str | None:
    """Return the AWS region of the current EC2 instance, or None if not on EC2.

    Queries the IMDSv2 endpoint (169.254.169.254), which is only reachable from
    within an EC2 instance.  The 1-second timeout makes this a no-op on any
    other machine.  Results are cached so the network round-trip happens at most
    once per process.
    """
    try:
        token_req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        )
        with urllib.request.urlopen(token_req, timeout=1) as resp:
            token = resp.read().decode()
        region_req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/placement/region",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with urllib.request.urlopen(region_req, timeout=1) as resp:
            return resp.read().decode()
    except Exception:
        return None


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
    lora_config: LoraAdapterConfig,
) -> torch.nn.Module:
    """
    Applies PEFT LoRA adapters to a model.

    Args:
        model: The model to apply LoRA to.
        lora_config: A LoraAdapterConfig instance (from workshop_infrastructure.configs).

    Returns:
        Model with PEFT LoRA adapters applied.
    """
    print(
        f"Applying PEFT LoRA: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
        f"dropout={lora_config.lora_dropout}, modules={lora_config.target_modules}"
    )

    from peft import LoraConfig, get_peft_model
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        target_modules=lora_config.target_modules,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
    )

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


def load_pretrained_weights(model: torch.nn.Module, pretrained_path: Optional[str]) -> None:
    """Load pretrained weights into a fine-tuning model, skipping shape-mismatched keys.

    The pretrained checkpoint was saved from HelioSpectFormer directly, so its keys
    are flat (e.g. ``embedding.proj.weight``).  The composed fine-tuning models
    (HelioSpectformer1D, HelioSpectformer2D) nest the backbone under ``backbone.*``,
    so we try both the original key and the ``backbone.``-prefixed key when matching
    against the current model's state dict.

    Args:
        model: The fine-tuning model to load weights into.
        pretrained_path: Path to the pretrained checkpoint (.pt file). No-op if None.
    """
    if not pretrained_path:
        return
    print(f"Loading pretrained weights from {pretrained_path}.")
    model_state = model.state_dict()
    checkpoint_state = torch.load(pretrained_path, weights_only=True, map_location="cpu")

    remapped = {}
    for k, v in checkpoint_state.items():
        for candidate in (k, f"backbone.{k}"):
            if candidate in model_state and hasattr(v, "shape") and v.shape == model_state[candidate].shape:
                remapped[candidate] = v
                break

    model_state.update(remapped)
    model.load_state_dict(model_state, strict=True)
    print(f"Loaded {len(remapped)} / {len(checkpoint_state)} pretrained weights.")


class UploadBestCheckpointToS3(L.Callback):
    """Lightning callback that uploads the best checkpoint to S3 after each validation epoch.

    - No-op unless ``bucket`` is set (mirrors the ``output.s3_bucket: null`` YAML default).
    - Only runs from global rank 0 under DDP to avoid duplicate uploads.
    - Uses a stable S3 key (``fixed_key_name``) so the latest best is always at a
      predictable location regardless of epoch number.

    Args:
        checkpoint_cb: The ModelCheckpoint callback whose ``best_model_path`` to watch.
        bucket: S3 bucket name. Pass ``None`` to disable uploads entirely.
        prefix: Key prefix (folder) within the bucket, e.g. ``"flare/exp_001"``.
        fixed_key_name: Object name within the prefix. Defaults to ``"best.ckpt"``.
    """

    def __init__(
        self,
        checkpoint_cb: ModelCheckpoint,
        bucket: Optional[str],
        prefix: str = "",
        fixed_key_name: Optional[str] = "best.ckpt",
    ):
        super().__init__()
        self.checkpoint_cb = checkpoint_cb
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.fixed_key_name = fixed_key_name
        self.last_uploaded_best = None
        self._s3 = None

    def _upload_if_new_best(self, trainer) -> None:
        if not self.bucket:
            return
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        best_path = getattr(self.checkpoint_cb, "best_model_path", None)
        if not best_path or best_path == self.last_uploaded_best:
            return

        ckpt_path = Path(best_path)
        if not ckpt_path.exists():
            return

        if self._s3 is None:
            try:
                import boto3
            except ImportError as e:
                raise RuntimeError(
                    "boto3 is required for S3 uploads. Install with: pip install boto3"
                ) from e
            self._s3 = boto3.client("s3")

        object_name = self.fixed_key_name or ckpt_path.name
        s3_key = f"{self.prefix}/{object_name}" if self.prefix else object_name

        print(f"[S3] Uploading {ckpt_path} -> s3://{self.bucket}/{s3_key}")
        self._s3.upload_file(str(ckpt_path), self.bucket, s3_key)
        print("[S3] Upload complete.")
        self.last_uploaded_best = str(ckpt_path)

    def on_validation_end(self, trainer, pl_module) -> None:
        self._upload_if_new_best(trainer)

    def on_fit_end(self, trainer, pl_module) -> None:
        # Final check in case the last best update happens near fit end.
        self._upload_if_new_best(trainer)