"""
Typed configuration for Surya downstream applications.

Everything a fine-tuning run needs is declared here:

- ``TimeEmbeddingConfig`` / ``LoraAdapterConfig`` / ``ModelConfig`` — the backbone
  architecture and fine-tuning head, mirroring the HelioSpectformer constructors.
- ``DataConfig`` / ``OutputConfig`` / ``TrainingConfig`` — the parts of a run that every
  downstream task has: what data to load, where checkpoints go, and the optimizer /
  logging settings.
- ``load_config()`` — the single entry point. Reads the YAML and returns a fully typed
  ``TrainingConfig``, so downstream code never passes raw dicts around.

**Adding task-specific settings:** subclass ``DataConfig`` in your app and pass the
subclass to ``load_config(path, data_cls=YourDataConfig)``. You do not copy this file.
See ``downstream_apps/template/configs.py`` for a worked example.
"""

from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field, fields as dc_fields
from pathlib import Path
from typing import ClassVar, List, Optional, Union

import yaml

# Environment variable consulted when data.s3_cache_dir is left unset in the config.
# It lets one committed config run on machines with different scratch space, without an
# absolute path pointing into one user's home ending up in version control.
S3_CACHE_DIR_ENV_VAR = "SURYA_WS_CACHE_DIR"


# How HelioNetCDFDataset reads s3:// paths. Declared here rather than in helio.py so the
# config layer can reject a bad value at startup, before any dataset is constructed.
#   "download"    — fetch the whole object into s3_cache_dir, then open it locally.
#                   Recommended: NetCDF/HDF5 needs random seeks that streaming cannot serve.
#   "simplecache" — fsspec read-through cache into s3_cache_dir.
#   "stream"      — direct s3fs handle, no local cache. Works, but roughly 9x slower on
#                   full SDO frames: HDF5 random access becomes many small ranged GETs.
VALID_S3_MODES = ("download", "simplecache", "stream")

# Accepted values for training.deterministic, passed straight to lightning.Trainer.
#   False   — no determinism guarantees; two identical runs may differ. The default,
#             chosen for throughput: determinism costs roughly 20% of wall time.
#   "warn"  — bit-identical wherever a deterministic kernel exists; warn (naming the op)
#             where one does not. Never blocks a run. Switch to this whenever you need to
#             attribute a change in results to your edit rather than to run-to-run drift.
#   True    — hard guarantee; any op without a deterministic kernel raises. Note that
#             model.learned_flow uses F.grid_sample, which has no deterministic CUDA
#             backward, so learned_flow: true is incompatible with this setting.
VALID_DETERMINISTIC = (True, False, "warn")


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

    # --- Checkpoint ---
    # Path to the pretrained Surya backbone weights. Passed to load_pretrained_weights().
    # Kept on ModelConfig (not TrainingConfig) because it describes the model, not the run.
    pretrained_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """What data to load, how to sample it in time, and how to reach it on S3.

    Task-specific fields (label catalogs, alignment parameters, …) do not belong here.
    Subclass this in your app and pass the subclass to ``load_config(data_cls=...)``.
    """

    # Fields holding a filesystem path. Relative values are resolved against the config
    # file's own directory, so a checked-in config works from any working directory and
    # on any machine — no absolute paths pointing into one user's home.
    # A subclass that adds a path field must extend this, e.g.:
    #     PATH_FIELDS = DataConfig.PATH_FIELDS + ("my_catalog_path",)
    PATH_FIELDS: ClassVar[tuple[str, ...]] = (
        "train_data_path",
        "valid_data_path",
        "scalers_path",
        "sdo_data_root_path",
    )

    # --- Paths (relative paths are resolved against the config file's directory) ---
    train_data_path: str
    valid_data_path: str
    scalers_path: str

    # --- Input channels and temporal sampling ---
    channels: List[str]
    time_delta_input_minutes: List[int]
    time_delta_target_minutes: int

    # --- Local data ---
    # Root prepended to relative local paths in the index. Ignored for s3:// paths.
    sdo_data_root_path: Optional[str] = None

    # --- S3 (used only when the index contains s3:// paths) ---
    s3_anon: bool = False
    # One of VALID_S3_MODES (above). "download" is recommended for NetCDF/HDF5.
    s3_mode: str = "download"
    # Required unless s3_mode is "stream". Budget ~1 GB per unique timestep.
    # Left unset here, load_config() falls back to $SURYA_WS_CACHE_DIR.
    s3_cache_dir: Optional[str] = None
    s3_boto3_max_concurrency: int = 4   # parallel threads per multipart download
    s3_boto3_part_size_mb: int = 64     # part size in MB for multipart downloads

    # --- Development ---
    max_samples: Optional[int] = None

    def __post_init__(self) -> None:
        if self.s3_mode not in VALID_S3_MODES:
            raise ValueError(
                f"Unknown data.s3_mode {self.s3_mode!r}. "
                f"Valid modes are: {', '.join(VALID_S3_MODES)}."
            )


@dataclass
class OutputConfig:
    """Paths and S3 settings for checkpoints and artifacts."""
    ckpt_dir: str = "checkpoints"
    # S3 upload of best checkpoint (all three used together; leave s3_bucket null to disable)
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
    num_workers: int = 8
    # Seeds every RNG (Python, NumPy, torch, and each DataLoader worker) and the shuffle
    # order. This applies regardless of `deterministic`: even with determinism off, the
    # data order is pinned, so changing the seed is how you get a genuinely different run.
    seed: int = 42
    # One of VALID_DETERMINISTIC (above). Defaults to False for speed; set "warn" when
    # you need reproducible results.
    deterministic: Union[bool, str] = False
    rollout_steps: int = 0
    drop_hmi_probability: float = 0.0
    use_latitude_in_learned_flow: bool = False
    dtype: str = "float32"
    wandb_project: str = "surya_downstream"
    wandb_entity: Optional[str] = None

    def __post_init__(self) -> None:
        # Normalize the string form so YAML "Warn"/"WARN" behave like "warn"; bools pass
        # through untouched (YAML true/false already parse as Python bools).
        if isinstance(self.deterministic, str):
            self.deterministic = self.deterministic.lower()
        if self.deterministic not in VALID_DETERMINISTIC:
            raise ValueError(
                f"Unknown training.deterministic {self.deterministic!r}. "
                "Valid values are: true, false, \"warn\"."
            )
        # A quoted seed in YAML ("42") would otherwise reach torch.manual_seed as a str
        # and fail deep in the training loop. bool is excluded because it subclasses int.
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(
                f"training.seed must be an integer, got {self.seed!r} "
                f"({type(self.seed).__name__}). Write it unquoted in the YAML, e.g. seed: 42."
            )


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------

def _from_dict(cls, d: dict, section: str):
    """Construct a dataclass from a dict, rejecting unrecognized keys.

    Unknown keys are an error, not a silent no-op. A dropped key looks exactly like a
    working one: adding ``s3_mode: stream`` to the YAML without a matching dataclass
    field would otherwise change nothing and report nothing. Failing here names the
    typo and lists what was expected.
    """
    if not isinstance(d, dict):
        raise ValueError(f"Config section '{section}:' must be a mapping, got {type(d).__name__}.")

    known = {f.name for f in dc_fields(cls)}
    unknown = sorted(set(d) - known)
    if unknown:
        raise ValueError(
            f"Unrecognized key(s) in '{section}:' section of the config: "
            f"{', '.join(unknown)}.\n"
            f"Valid keys for {cls.__name__} are: {', '.join(sorted(known))}.\n"
            "If this is a new task-specific setting, add it to your DataConfig subclass "
            "(see downstream_apps/template/configs.py)."
        )

    missing = [
        f.name for f in dc_fields(cls)
        if f.name not in d
        and f.default is MISSING
        and f.default_factory is MISSING  # type: ignore[misc]
    ]
    if missing:
        raise ValueError(
            f"Missing required key(s) in '{section}:' section of the config: "
            f"{', '.join(missing)}."
        )

    return cls(**d)


def _resolve_path(value, config_dir: Path) -> str:
    """Expand ``~``/``$VARS`` and anchor a relative path to the config file's directory."""
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if not os.path.isabs(expanded):
        expanded = str((config_dir / expanded).resolve())
    return expanded


def _resolve_paths(data_cfg, config_dir: Path) -> None:
    """Expand ``~``/env vars and resolve relative path fields against the config's directory."""
    for name in getattr(type(data_cfg), "PATH_FIELDS", DataConfig.PATH_FIELDS):
        value = getattr(data_cfg, name, None)
        if value:
            setattr(data_cfg, name, _resolve_path(value, config_dir))

    # s3_cache_dir is a scratch location, not a repo path. Precedence is
    # CLI flag > config value > $SURYA_WS_CACHE_DIR; expand ~ and $VARS, but never anchor
    # it to the config directory (that would put ~1 GB files inside the repo).
    cache_dir = getattr(data_cfg, "s3_cache_dir", None) or os.environ.get(S3_CACHE_DIR_ENV_VAR)
    if cache_dir:
        data_cfg.s3_cache_dir = os.path.expandvars(os.path.expanduser(str(cache_dir)))


def load_config(
    path: Union[str, Path],
    data_cls: type = DataConfig,
) -> TrainingConfig:
    """Parse a run config YAML into a typed ``TrainingConfig``.

    The YAML has five top-level sections (data, model, training, output, logging) plus
    ``job_id``. Each is parsed into its dataclass; unknown keys raise rather than being
    silently dropped.

    Args:
        path: Path to the config YAML.
        data_cls: The ``DataConfig`` subclass to parse the ``data:`` section into. Pass
            your app's subclass to accept task-specific keys.

    Returns:
        A fully typed ``TrainingConfig``. Relative paths in the ``data:`` section are
        resolved against the config file's own directory.
    """
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config {config_path} must contain a top-level mapping.")

    missing_sections = [k for k in ("job_id", "data", "model") if k not in raw]
    if missing_sections:
        raise ValueError(
            f"Config {config_path} is missing required top-level key(s): "
            f"{', '.join(missing_sections)}. Expected: job_id, data, model "
            "(training, output and logging are optional)."
        )

    # model: build the nested configs first, then the ModelConfig around them.
    model_raw = dict(raw["model"])
    model_raw["time_embedding"] = _from_dict(
        TimeEmbeddingConfig, model_raw.get("time_embedding", {}), "model.time_embedding"
    )
    model_raw["lora_config"] = _from_dict(
        LoraAdapterConfig, model_raw.get("lora_config", {}), "model.lora_config"
    )
    model_cfg = _from_dict(ModelConfig, model_raw, "model")

    # pretrained_path is a path like the data: ones, and gets the same treatment.
    if model_cfg.pretrained_path:
        model_cfg.pretrained_path = _resolve_path(model_cfg.pretrained_path, config_path.parent)

    data_cfg = _from_dict(data_cls, raw["data"], "data")
    _resolve_paths(data_cfg, config_path.parent)

    training = raw.get("training", {}) or {}
    logging_cfg = raw.get("logging", {}) or {}
    _check_section_keys(training, _TRAINING_KEYS, "training")
    _check_section_keys(logging_cfg, _LOGGING_KEYS, "logging")

    return TrainingConfig(
        job_id=raw["job_id"],
        data=data_cfg,
        model=model_cfg,
        output=_from_dict(OutputConfig, raw.get("output", {}) or {}, "output"),
        learning_rate=training.get("learning_rate", 1e-4),
        max_epochs=training.get("max_epochs", 20),
        batch_size=training.get("batch_size", 2),
        num_workers=training.get("num_workers", 8),
        seed=training.get("seed", 42),
        deterministic=training.get("deterministic", False),
        rollout_steps=training.get("rollout_steps", 0),
        drop_hmi_probability=training.get("drop_hmi_probability", 0.0),
        use_latitude_in_learned_flow=training.get("use_latitude_in_learned_flow", False),
        dtype=training.get("dtype", "float32"),
        wandb_project=logging_cfg.get("wandb_project", "surya_downstream"),
        wandb_entity=logging_cfg.get("wandb_entity"),
    )


# The training: and logging: sections map onto flat TrainingConfig fields rather than
# onto a dataclass of their own, so their keys are checked explicitly.
_TRAINING_KEYS = frozenset({
    "learning_rate", "max_epochs", "batch_size", "num_workers", "seed", "deterministic",
    "rollout_steps", "drop_hmi_probability", "use_latitude_in_learned_flow", "dtype",
})
_LOGGING_KEYS = frozenset({"wandb_project", "wandb_entity"})


def _check_section_keys(section: dict, valid: frozenset, name: str) -> None:
    """Raise on unrecognized keys in a section that has no dataclass of its own."""
    unknown = sorted(set(section) - valid)
    if unknown:
        raise ValueError(
            f"Unrecognized key(s) in '{name}:' section of the config: {', '.join(unknown)}.\n"
            f"Valid keys are: {', '.join(sorted(valid))}."
        )
