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

# Optional: S3 helpers below degrade to a clear ImportError when boto3 is absent.
try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig
except Exception:  # pragma: no cover
    boto3 = None
    UNSIGNED = None
    BotoConfig = None


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


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``.

    Raises:
        ValueError: If ``uri`` is not an S3 URI, or has no key component.
    """
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got: {uri!r}")
    remainder = uri[len("s3://"):]
    if "/" not in remainder:
        raise ValueError(f"S3 URI is missing a key: {uri!r} (expected s3://bucket/key)")
    bucket, key = remainder.split("/", 1)
    return bucket, key


def make_s3_client(
    anon: bool = False,
    region: str | None = None,
    pool_size: int = 32,
    max_attempts: int = 10,
):
    """Return a configured boto3 S3 client.

    Shared by the dataset loader and the S3 benchmark so both use identical connection
    pooling and retry behaviour.

    Args:
        anon: If True, sign requests anonymously (public buckets).
        region: AWS region. Pass ``None`` to let boto3 resolve it.
        pool_size: Max connections in the client's connection pool. Should be at least
            twice the download concurrency, or threads will contend for connections.
        max_attempts: Retry attempts in adaptive mode. The dataset uses a high value
            because a failed read kills a training run; the benchmark uses a low one so
            throughput measurements are not skewed by retries.
    """
    if boto3 is None:
        raise ImportError("boto3 is required for S3 access. Install via: pip install boto3")

    retry_cfg = {"max_attempts": max_attempts, "mode": "adaptive"}
    if anon:
        config = BotoConfig(
            signature_version=UNSIGNED,
            max_pool_connections=pool_size,
            retries=retry_cfg,
        )
    else:
        config = BotoConfig(max_pool_connections=pool_size, retries=retry_cfg)
    return boto3.client("s3", region_name=region, config=config)


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
        info: Path to a scalers YAML file, or an already-loaded dict mapping
            channel name -> scaler parameters.

    Returns:
        Dict mapping channel name -> scaler instance (e.g. ``StandardScaler``), each
        exposing ``.mean``, ``.std``, ``.epsilon`` and ``.sl_scale_factor``.

    On the ``base:`` field
    ---------------------
    Every entry records the module the scaler was originally fitted in — in the
    shipped ``scalers.yaml`` that is ``surya.datasets.transformations``, which does not
    exist in this repo (the Surya code is vendored under ``workshop_infrastructure/``).

    That field is deliberately **ignored**. Classes are always resolved from
    ``workshop_infrastructure.datasets.transformations``, never imported from the
    recorded path, so normalization does not depend on what happens to be installed in
    the environment: were a real ``surya`` package ever added, honoring ``base`` would
    silently switch which ``StandardScaler`` implementation runs, and results could move
    because of an unrelated install. Determinism matters more here than deference to a
    stale field. This is not warned about at runtime — the field is stale in every entry
    of a file fetched from HuggingFace that nobody can edit, so a warning would fire
    always and invite no action.
    """
    import yaml
    import workshop_infrastructure.datasets.transformations as _transformations

    source = "<dict>"
    if not isinstance(info, dict):
        source = str(info)
        if not os.path.isfile(source):
            raise FileNotFoundError(
                f"Scalers file not found: {source}\n"
                "This is the data.scalers_path entry in your config. It is downloaded "
                "automatically on the first run; to fetch it explicitly:\n"
                "    python -m workshop_infrastructure.assets --scalers --dest <assets dir>"
            )
        with open(source, "r", encoding="utf-8") as f:
            info = yaml.safe_load(f)

    if not isinstance(info, dict):
        raise ValueError(f"Scalers source {source} must contain a mapping of channel name -> parameters.")

    available = sorted(
        name for name in dir(_transformations)
        if isinstance(getattr(_transformations, name), type)
    )

    ret_dict = {}
    for p_key, p_val in info.items():
        if not isinstance(p_val, dict) or "class" not in p_val:
            raise ValueError(
                f"Scalers entry {p_key!r} in {source} is malformed: expected a mapping "
                f"with a 'class' key, got {type(p_val).__name__}."
            )
        class_name = p_val["class"]
        # Note: p_val["base"] is intentionally not consulted — see the docstring.
        if not hasattr(_transformations, class_name):
            raise ValueError(
                f"Scalers entry {p_key!r} in {source} names class {class_name!r}, which does "
                "not exist in workshop_infrastructure.datasets.transformations.\n"
                f"Available classes: {', '.join(available)}."
            )
        ret_dict[p_key] = getattr(_transformations, class_name).from_dict(p_val)
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