import os
import re
import random
import hashlib
from datetime import datetime
from typing import Tuple
import torch
import numpy as np
import skimage.measure
import xarray as xr
import pandas as pd
from logging import Logger
from torch.utils.data import Dataset
from workshop_infrastructure.utils import get_rank, create_logger
from functools import cache

# Optional S3 support via fsspec/s3fs (read-through streaming)
try:
    import fsspec
    import s3fs
except Exception:  # pragma: no cover
    fsspec = None
    s3fs = None

# Optional S3 support via boto3 (recommended: faster whole-object downloads)
try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig
    from boto3.s3.transfer import TransferConfig
except Exception:  # pragma: no cover
    boto3 = None
    UNSIGNED = None
    BotoConfig = None
    TransferConfig = None

from numba import njit, prange
import hdf5plugin


# ---------------------------------------------------------------------------
# Signum-log transforms (module-level so numba can JIT-compile them)
# ---------------------------------------------------------------------------

@njit(parallel=True)
def fast_transform(data, means, stds, sl_scale_factors, epsilons):
    """
    Signum-log normalization using Numba for speed.

    Notes:
    - Must reside outside class definitions (Numba requirement).
    - May cause hangs on some GPU clusters when called from dataloader workers;
      use the pure-NumPy ``transform`` function in that case.

    Args:
        data: Numpy array of shape (C, H, W).
        means: Per-channel means, shape (C,).
        stds: Per-channel standard deviations, shape (C,).
        sl_scale_factors: Per-channel signum-log scale factors, shape (C,).
        epsilons: Per-channel small constants to avoid zero division, shape (C,).

    Returns:
        Normalized array of shape (C, H, W), dtype float32.
    """
    C, H, W = data.shape
    out = np.empty((C, H, W), dtype=np.float32)
    for c in prange(C):
        mean = means[c]
        std = stds[c]
        eps = epsilons[c]
        sl_scale_factor = sl_scale_factors[c]
        for i in range(H):
            for j in range(W):
                val = data[c, i, j] * sl_scale_factor
                val = np.log1p(val) if val >= 0 else -np.log1p(-val)
                out[c, i, j] = (val - mean) / (std + eps)
    return out


@njit(parallel=True)
def fast_inverse_transform(data, means, stds, sl_scale_factors, epsilons):
    """
    Inverse signum-log normalization using Numba for speed.

    Args:
        data: Normalized array of shape (C, H, W).
        means: Per-channel means, shape (C,).
        stds: Per-channel standard deviations, shape (C,).
        sl_scale_factors: Per-channel signum-log scale factors, shape (C,).
        epsilons: Per-channel small constants to avoid zero division, shape (C,).

    Returns:
        Reconstructed array of shape (C, H, W), dtype float32.
    """
    C, H, W = data.shape
    out = np.empty((C, H, W), dtype=np.float32)
    for c in prange(C):
        mean = means[c]
        std = stds[c]
        eps = epsilons[c]
        sl_scale_factor = sl_scale_factors[c]
        for i in range(H):
            for j in range(W):
                val = data[c, i, j] * (std + eps) + mean
                val = np.expm1(val) if val >= 0 else -np.expm1(-val)
                out[c, i, j] = val / sl_scale_factor
    return out


def transform(
    data: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    sl_scale_factors: np.ndarray,
    epsilons: np.ndarray,
) -> np.ndarray:
    """
    Signum-log normalization (pure NumPy). Drop-in replacement for ``fast_transform``.

    Args:
        data: Numpy array of shape (C, H, W).
        means: Per-channel means, shape (C,).
        stds: Per-channel standard deviations, shape (C,).
        sl_scale_factors: Per-channel signum-log scale factors, shape (C,).
        epsilons: Per-channel small constants to avoid zero division, shape (C,).

    Returns:
        Normalized array of shape (C, H, W).
    """
    means = means.reshape(*means.shape, 1, 1)
    stds = stds.reshape(*stds.shape, 1, 1)
    sl_scale_factors = sl_scale_factors.reshape(*sl_scale_factors.shape, 1, 1)
    epsilons = epsilons.reshape(*epsilons.shape, 1, 1)

    data = data * sl_scale_factors
    data = np.sign(data) * np.log1p(np.abs(data))
    data = (data - means) / (stds + epsilons)
    return data


def inverse_transform_single_channel(data, mean, std, sl_scale_factor, epsilon):
    """
    Inverse signum-log normalization for a single channel.

    Args:
        data: Numpy array of shape (H, W).
        mean: Scalar mean.
        std: Scalar standard deviation.
        sl_scale_factor: Scalar signum-log scale factor.
        epsilon: Small constant to avoid zero division.

    Returns:
        Reconstructed array of shape (H, W).
    """
    data = data * (std + epsilon) + mean
    data = np.sign(data) * np.expm1(np.abs(data))
    return data / sl_scale_factor


# ---------------------------------------------------------------------------
# Channel masking transform
# ---------------------------------------------------------------------------

class RandomChannelMaskerTransform:
    """Randomly zero-out AIA channels and optionally the HMI channel.

    Used during pretraining to improve robustness to missing inputs.

    Args:
        num_channels: Total number of channels in the input tensor.
        num_mask_aia_channels: Number of AIA channels to randomly zero-out per sample.
        phase: Dataset phase (e.g., ``"train"``). Included for future phase-specific logic.
        drop_hmi_probability: Probability of zeroing the last (HMI) channel.
    """

    def __init__(self, num_channels, num_mask_aia_channels, phase, drop_hmi_probability):
        self.num_channels = num_channels
        self.num_mask_aia_channels = num_mask_aia_channels
        self.drop_hmi_probability = drop_hmi_probability

    def __call__(self, input_tensor):
        C, T, H, W = input_tensor.shape

        channels_to_mask = random.sample(range(C), self.num_mask_aia_channels)
        mask = torch.ones((C, 1, 1, 1))
        mask[channels_to_mask, ...] = 0
        masked_tensor = input_tensor * mask

        if self.drop_hmi_probability > random.random():
            masked_tensor[-1, ...] = 0

        return masked_tensor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HelioNetCDFDataset(Dataset):
    """
    PyTorch dataset for SDO NetCDF files, supporting both local and S3 storage.

    Loads multi-channel (AIA + HMI) solar image stacks and pairs them with
    forecast targets for temporal prediction tasks. Handles variable input
    timestep sampling and is compatible with downstream fine-tuning.

    Index format (CSV with columns: timestep, path, present):

        timestep                    path                                present
        2011-01-01 00:00:00  s3://bucket/2011/01/sample.nc              1
        2011-01-01 00:12:00  s3://bucket/2011/01/sample.nc              1
        ...

    Valid samples are timesteps for which all required input and target
    offsets can be resolved in the index.

    Args:
        index_path: Path to the CSV index file.
        time_delta_input_minutes: List of input time offsets in minutes from the reference timestep.
        time_delta_target_minutes: Target time step in minutes; rollout targets are multiples of this.
        n_input_timestamps: Number of input frames to sample.
        rollout_steps: Number of forecast steps.
        scalers: Per-channel normalization statistics (see ``workshop_infrastructure/configs.py``).
        num_mask_aia_channels: Number of AIA channels to randomly mask during training.
        drop_hmi_probability: Probability of dropping the HMI channel during training.
        use_latitude_in_learned_flow: If True, include heliographic latitude in the output dict.
        channels: List of NetCDF variable names to load. Defaults to 8 AIA + HMI channels.
        phase: Dataset phase label (e.g., ``"train"``, ``"val"``).
        pooling: Spatial downsampling factor (average pooling). ``None`` or ``1`` means no pooling.
        random_vert_flip: If True, randomly flip images vertically during training.
        sdo_data_root_path: Optional root directory prepended to relative local paths.
        s3_storage_options: Options forwarded to fsspec/s3fs (e.g., ``{'anon': True}`` for public buckets).
        s3_use_simplecache: If True, use fsspec simplecache for read-through S3 caching.
        s3_cache_dir: Local directory for the S3 cache. Defaults to ``/tmp/helio_s3_cache``.
        s3fs_kwargs: Additional kwargs passed to ``s3fs.S3FileSystem``.
        s3_download_to_temp: If True (recommended for NetCDF/HDF5), download each S3 object to a
            local file before opening. Avoids seekability issues with streaming reads.
        s3_temp_dir: Directory for downloaded S3 files. Defaults to ``s3_cache_dir``.
        s3_boto3_max_concurrency: Number of parallel threads for boto3 multipart downloads.
        s3_boto3_part_size_mb: Part size in MB for boto3 multipart downloads.
    """

    def __init__(
        self,
        index_path: str,
        time_delta_input_minutes: list[int],
        time_delta_target_minutes: int,
        n_input_timestamps: int,
        rollout_steps: int,
        scalers=None,
        num_mask_aia_channels: int = 0,
        drop_hmi_probability: float = 0.0,
        use_latitude_in_learned_flow: bool = False,
        channels: list[str] | None = None,
        phase: str = "train",
        pooling: int | None = None,
        random_vert_flip: bool = False,
        sdo_data_root_path: str | None = None,
        # S3 options (only used when index contains s3:// URIs)
        s3_storage_options: dict | None = None,
        s3_use_simplecache: bool = False,
        s3_cache_dir: str = "/tmp/helio_s3_cache",
        s3fs_kwargs: dict | None = None,
        s3_download_to_temp: bool = True,
        s3_temp_dir: str | None = None,
        s3_boto3_max_concurrency: int = 4,
        s3_boto3_part_size_mb: int = 64,
    ):
        self.scalers = scalers
        self.phase = phase
        self.num_mask_aia_channels = num_mask_aia_channels
        self.drop_hmi_probability = drop_hmi_probability
        self.n_input_timestamps = n_input_timestamps
        self.rollout_steps = rollout_steps
        self.use_latitude_in_learned_flow = use_latitude_in_learned_flow
        self.pooling = pooling if pooling is not None else 1
        self.random_vert_flip = random_vert_flip
        self.sdo_data_root_path = sdo_data_root_path

        self.s3_storage_options = s3_storage_options or {}
        self.s3_use_simplecache = s3_use_simplecache
        self.s3_cache_dir = s3_cache_dir
        self.s3fs_kwargs = s3fs_kwargs or {}
        self.s3_download_to_temp = s3_download_to_temp
        self.s3_temp_dir = s3_temp_dir if s3_temp_dir is not None else s3_cache_dir
        self.s3_boto3_max_concurrency = s3_boto3_max_concurrency
        self.s3_boto3_part_size_mb = s3_boto3_part_size_mb
        self._s3fs = None  # lazily initialized per process

        self.channels = channels if channels is not None else [
            "0094", "0131", "0171", "0193", "0211", "0304", "0335", "hmi"
        ]
        self.in_channels = len(self.channels)

        self.masker = RandomChannelMaskerTransform(
            num_channels=self.in_channels,
            num_mask_aia_channels=self.num_mask_aia_channels,
            phase=self.phase,
            drop_hmi_probability=self.drop_hmi_probability,
        )

        self.time_delta_input_minutes = sorted(
            np.timedelta64(t, "m") for t in time_delta_input_minutes
        )
        self.time_delta_target_minutes = [
            np.timedelta64(iroll * time_delta_target_minutes, "m")
            for iroll in range(1, rollout_steps + 2)
        ]

        self.index = pd.read_csv(index_path)
        self.index = self.index[self.index["present"] == 1]
        self.index["timestep"] = pd.to_datetime(self.index["timestep"]).values.astype("datetime64[ns]")
        self.index.set_index("timestep", inplace=True)
        self.index.sort_index(inplace=True)

        self.valid_indices = self._filter_valid_indices()
        self.adjusted_length = len(self.valid_indices)

        self.rank = get_rank()
        self.logger: Logger | None = None

    # ------------------------------------------------------------------
    # Index filtering
    # ------------------------------------------------------------------

    def _filter_valid_indices(self) -> list:
        """Return the list of reference timesteps for which all required offsets are present."""
        time_deltas = np.unique(self.time_delta_input_minutes + self.time_delta_target_minutes)
        return [
            ts for ts in self.index.index
            if all(ts + dt in self.index.index for dt in time_deltas)
        ]

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _ensure_logger(self):
        """Create a per-process logger on first use."""
        if self.logger is not None:
            return
        os.makedirs("logs/data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        pid = os.getpid()
        self.logger = create_logger(
            output_dir="logs/data",
            dist_rank=self.rank,
            name=f"{timestamp}_{self.rank:>03}_data_{self.phase}_{pid}",
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Load and return a single sample.

        Args:
            idx: Dataset index.

        Returns:
            Dictionary with keys:
                ts (np.ndarray):               (C, T, H, W) — input frames
                time_delta_input (np.ndarray): (T,) — input time offsets in hours
                forecast (np.ndarray):         (C, L, H, W) — target frames
                lead_time_delta (np.ndarray):  (L,) — forecast lead times in hours
            When ``use_latitude_in_learned_flow=True``, also includes:
                input_latitudes (list[float])
                forecast_latitude (list[float])
        """
        self._ensure_logger()
        self.logger.info(f"Retrieving index {idx}.")
        return self._get_index_data(idx)

    # ------------------------------------------------------------------
    # Internal data loading
    # ------------------------------------------------------------------

    def _get_index_data(self, idx: int) -> dict:
        time_deltas = np.array(
            sorted(random.sample(self.time_delta_input_minutes[:-1], self.n_input_timestamps - 1))
            + [self.time_delta_input_minutes[-1]]
            + self.time_delta_target_minutes
        )
        reference_timestep = self.valid_indices[idx]
        required_timesteps = reference_timestep + time_deltas

        sequence_data = [
            self.transform_data(self.load_nc_data(self.index.loc[ts, "path"], ts, self.channels))
            for ts in required_timesteps
        ]

        inputs = sequence_data[: -self.rollout_steps - 1]
        targets = sequence_data[-self.rollout_steps - 1:]

        stacked_inputs = np.stack(inputs, axis=1)
        stacked_targets = np.stack(targets, axis=1)

        if self.num_mask_aia_channels > 0 or self.drop_hmi_probability:
            stacked_inputs = self.masker(stacked_inputs)

        if self.random_vert_flip and torch.bernoulli(torch.ones(()) / 2) == 1:
            stacked_inputs = torch.flip(stacked_inputs, dims=[-2])
            stacked_targets = torch.flip(stacked_targets, dims=[-2])

        time_delta_input_float = (
            (time_deltas[-self.rollout_steps - 2] - time_deltas[: -self.rollout_steps - 1])
            / np.timedelta64(1, "h")
        ).astype(np.float32)

        lead_time_delta_float = (
            (time_deltas[-self.rollout_steps - 2] - time_deltas[-self.rollout_steps - 1:])
            / np.timedelta64(1, "h")
        ).astype(np.float32)

        sample = {
            "ts": stacked_inputs,
            "time_delta_input": time_delta_input_float,
            "forecast": stacked_targets,
            "lead_time_delta": lead_time_delta_float,
        }

        if self.use_latitude_in_learned_flow:
            from sunpy.coordinates.ephemeris import get_earth

            latitudes = [get_earth(ts).lat.value for ts in required_timesteps]
            sample["input_latitudes"] = latitudes[: -self.rollout_steps - 1]
            sample["forecast_latitude"] = latitudes[-self.rollout_steps - 1:]

        return sample

    # ------------------------------------------------------------------
    # NetCDF loading
    # ------------------------------------------------------------------

    def load_nc_data(self, filepath: str, timestep: pd.Timestamp, channels: list[str]) -> np.ndarray:
        """
        Load a NetCDF file and return channel-stacked data as a NumPy array.

        Supports both local filesystem paths and S3 URIs (``s3://bucket/key``).
        When loading from S3, files are downloaded to a local cache directory
        before opening (controlled by ``s3_download_to_temp``).

        Args:
            filepath: Local path or S3 URI.
            timestep: Timestamp for the file (used for logging).
            channels: List of variable names to extract, stacked into (C, H, W).

        Returns:
            NumPy array of shape (C, H, W).
        """
        self.logger.info(f"Reading file {filepath}.")

        if not self._is_s3_path(filepath) and self.sdo_data_root_path and not os.path.isabs(filepath):
            filepath = os.path.join(self.sdo_data_root_path, filepath)

        if self._is_s3_path(filepath):
            return self._load_s3_nc_data(filepath, channels)

        with xr.open_dataset(filepath, engine="h5netcdf", chunks=None, cache=False) as ds:
            return ds[channels].to_array().load().to_numpy()

    def _load_s3_nc_data(self, s3_uri: str, channels: list[str]) -> np.ndarray:
        """Download an S3 NetCDF file to local cache (if needed) and open it."""
        if boto3 is None and fsspec is None:
            raise ImportError(
                "S3 support requires either 'boto3' or 'fsspec'+'s3fs'. "
                "Install via: pip install boto3  or  pip install s3fs fsspec"
            )

        if self.s3_download_to_temp:
            # Download whole object to a stable cache path, then open locally.
            # Atomic write (partial → rename) avoids corrupted cache files on crash.
            cache_path = self._s3_cache_path(s3_uri)
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

            if not os.path.exists(cache_path):
                tmp_path = cache_path + ".partial"
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                self._download_s3_object(s3_uri, tmp_path)
                os.replace(tmp_path, cache_path)

            with xr.open_dataset(cache_path, engine="h5netcdf", chunks=None, cache=False) as ds:
                return ds[channels].to_array().load().to_numpy()

        # Streaming fallback (read-through via fsspec simplecache or direct s3fs)
        if fsspec is None:
            raise ImportError(
                "Streaming S3 reads require 'fsspec' and 's3fs'. "
                "Install via: pip install s3fs fsspec"
            )

        if self.s3_use_simplecache:
            s3_options = {**self.s3_storage_options, **self.s3fs_kwargs}
            opener = fsspec.open(
                f"simplecache::{s3_uri}",
                mode="rb",
                cache_storage=self.s3_cache_dir,
                s3=s3_options,
            )
        else:
            opener = self._get_s3fs().open(s3_uri, mode="rb")

        with opener as f:
            with xr.open_dataset(f, engine="h5netcdf", chunks=None, cache=False) as ds:
                return ds[channels].to_array().load().to_numpy()

    # ------------------------------------------------------------------
    # S3 helpers
    # ------------------------------------------------------------------

    def _is_s3_path(self, path: str) -> bool:
        return isinstance(path, str) and path.startswith("s3://")

    def _get_s3fs(self):
        """Lazily create and cache an ``s3fs.S3FileSystem`` instance (per process)."""
        if s3fs is None:
            raise ImportError("s3fs is required. Install via: pip install s3fs fsspec")
        if self._s3fs is None:
            self._s3fs = s3fs.S3FileSystem(**self.s3fs_kwargs, **self.s3_storage_options)
        return self._s3fs

    def _parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        """Split ``s3://bucket/key`` into ``(bucket, key)``."""
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Not an S3 URI: {s3_uri}")
        bucket, key = s3_uri[5:].split("/", 1)
        return bucket, key

    def _s3_cache_path(self, s3_uri: str) -> str:
        """Build a stable, human-readable local cache path for an S3 object."""
        bucket, key = self._parse_s3_uri(s3_uri)
        base = os.path.basename(key) or "object"
        base_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-") or "object"
        _, ext = os.path.splitext(base_safe)
        suffix = ext if ext else ".nc"
        key_hash = hashlib.sha1(f"{bucket}/{key}".encode()).hexdigest()
        fname = f"{bucket}__{key_hash}__{base_safe}"
        if not fname.endswith(suffix):
            fname += suffix
        return os.path.join(self.s3_temp_dir, fname)

    def _download_s3_object(self, s3_uri: str, local_path: str) -> None:
        """Download an S3 object to ``local_path``.

        Uses boto3's transfer manager when available (parallel multipart downloads),
        otherwise falls back to streaming via s3fs.
        """
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

        if boto3 is not None:
            anon = bool(self.s3_storage_options.get("anon") or self.s3fs_kwargs.get("anon"))
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            pool_size = max(32, self.s3_boto3_max_concurrency * 2)
            retry_config = {"max_attempts": 10, "mode": "adaptive"}

            if anon:
                client = boto3.client(
                    "s3",
                    region_name=region,
                    config=BotoConfig(
                        signature_version=UNSIGNED,
                        max_pool_connections=pool_size,
                        retries=retry_config,
                    ),
                )
            else:
                client = boto3.client(
                    "s3",
                    region_name=region,
                    config=BotoConfig(max_pool_connections=pool_size, retries=retry_config),
                )

            transfer_cfg = TransferConfig(
                multipart_threshold=self.s3_boto3_part_size_mb * 1024 * 1024,
                multipart_chunksize=self.s3_boto3_part_size_mb * 1024 * 1024,
                max_concurrency=self.s3_boto3_max_concurrency,
                use_threads=True,
                io_chunksize=1024 * 1024,
            )
            bucket, key = self._parse_s3_uri(s3_uri)
            client.download_file(bucket, key, local_path, Config=transfer_cfg)
            return

        # Fallback: stream via s3fs
        fs = self._get_s3fs()
        with fs.open(s3_uri, "rb") as src, open(local_path, "wb") as dst:
            for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                dst.write(chunk)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @cache
    def _transformation_inputs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        means = np.array([self.scalers[ch].mean for ch in self.channels])
        stds = np.array([self.scalers[ch].std for ch in self.channels])
        epsilons = np.array([self.scalers[ch].epsilon for ch in self.channels])
        sl_scale_factors = np.array([self.scalers[ch].sl_scale_factor for ch in self.channels])
        return means, stds, epsilons, sl_scale_factors

    def transform_data(self, data: np.ndarray) -> np.ndarray:
        """Apply signum-log normalization to a (C, H, W) array."""
        assert data.ndim == 3
        if self.pooling > 1:
            data = skimage.measure.block_reduce(data, block_size=(1, self.pooling, self.pooling), func=np.mean)
        means, stds, epsilons, sl_scale_factors = self._transformation_inputs()
        return transform(data, means, stds, sl_scale_factors, epsilons)

    def inverse_transform_data(self, data: np.ndarray) -> np.ndarray:
        """Invert signum-log normalization on a (C, H, W) array."""
        assert data.ndim == 3
        if self.pooling > 1:
            data = skimage.measure.block_reduce(data, block_size=(1, self.pooling, self.pooling), func=np.mean)
        means, stds, epsilons, sl_scale_factors = self._transformation_inputs()
        return fast_inverse_transform(data, means, stds, sl_scale_factors, epsilons)
