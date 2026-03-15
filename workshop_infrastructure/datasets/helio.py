"""
PyTorch dataset for SDO (Solar Dynamics Observatory) NetCDF files.

This module provides:
- ``HelioNetCDFDataset`` — the base dataset class used by all Surya downstream tasks.
  Handles index loading, temporal frame sampling, signum-log normalization, channel
  masking, and transparent local/S3 file access.
- ``RandomChannelMaskerTransform`` — callable that randomly zeros input channels to
  improve robustness to missing observations.
- Signum-log transform functions (``transform``, ``fast_transform``, and their inverses)
  used to normalize solar imagery before passing it to the model.

When writing a downstream task, subclass ``HelioNetCDFDataset`` and override
``__getitem__`` to attach your task-specific labels to the sample dict returned by
``super().__getitem__()``.
"""

import functools
import os
import re
import random
import hashlib
import urllib.request
import urllib.error
from datetime import datetime
import torch
import numpy as np
import skimage.measure
import xarray as xr
import pandas as pd
from logging import Logger
from torch.utils.data import Dataset
from workshop_infrastructure.utils import get_rank, create_logger

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


@functools.lru_cache(maxsize=1)
def _detect_ec2_region() -> str | None:
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
import hdf5plugin  # noqa: F401  # side-effect import: registers HDF5 compression filters


# ---------------------------------------------------------------------------
# Signum-log transforms (module-level so numba can JIT-compile them)
# ---------------------------------------------------------------------------
#
# Why signum-log?  Solar imagery (especially magnetograms and EUV channels)
# has values that span many orders of magnitude and are symmetric around zero.
# A plain log is undefined for negative values; a standard z-score squashes the
# large dynamic range unevenly.  Signum-log preserves sign, compresses extreme
# values, and then standardizes:
#
#   forward:  y = sign(x * s) * log1p(|x * s|)   then  (y - μ) / (σ + ε)
#   inverse:  y_raw = y * (σ + ε) + μ             then  sign(y_raw) * expm1(|y_raw|) / s
#
# where s = sl_scale_factor (per-channel amplitude scaling applied before the log).
# μ, σ, ε, and s are stored in the per-channel scaler objects built by
# ``workshop_infrastructure/utils.py:build_scalers()``.
#
# Two implementations are provided:
#   fast_transform / fast_inverse_transform — Numba JIT, parallel across channels.
#     Faster for large arrays but may hang on some GPU clusters with dataloader workers.
#   transform / inverse_transform_single_channel — pure NumPy, always safe.
# ---------------------------------------------------------------------------

@njit(parallel=True)
def fast_transform(data, means, stds, sl_scale_factors, epsilons):
    """Signum-log normalization, Numba parallel implementation.

    See the module-level comment for the mathematical definition.
    Must live outside class definitions (Numba requirement).
    May hang on some GPU clusters with dataloader workers — use ``transform`` in that case.

    Args:
        data: NumPy array of shape (C, H, W).
        means: Per-channel means, shape (C,).
        stds: Per-channel standard deviations, shape (C,).
        sl_scale_factors: Per-channel amplitude scaling factors, shape (C,).
        epsilons: Per-channel small constants that prevent division by zero, shape (C,).

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
    """Inverse signum-log normalization, Numba parallel implementation.

    See the module-level comment for the mathematical definition.

    Args:
        data: Normalized array of shape (C, H, W).
        means: Per-channel means, shape (C,).
        stds: Per-channel standard deviations, shape (C,).
        sl_scale_factors: Per-channel amplitude scaling factors, shape (C,).
        epsilons: Per-channel small constants that prevent division by zero, shape (C,).

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
    """Signum-log normalization, pure NumPy. Drop-in replacement for ``fast_transform``.

    Safe to use with DataLoader workers on any platform. See the module-level comment
    for the mathematical definition.

    Args:
        data: NumPy array of shape (C, H, W).
        means: Per-channel means, shape (C,).
        stds: Per-channel standard deviations, shape (C,).
        sl_scale_factors: Per-channel amplitude scaling factors, shape (C,).
        epsilons: Per-channel small constants that prevent division by zero, shape (C,).

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
    """Inverse signum-log normalization for a single channel (pure NumPy).

    Convenience wrapper for per-channel post-processing (e.g. model output visualization).
    For full (C, H, W) arrays use ``fast_inverse_transform`` or invert via
    ``HelioNetCDFDataset.inverse_transform_data()``.

    Args:
        data: NumPy array of shape (H, W).
        mean: Scalar mean.
        std: Scalar standard deviation.
        sl_scale_factor: Scalar amplitude scaling factor.
        epsilon: Small constant that prevents division by zero.

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
        """Apply random channel masking to a (C, T, H, W) image stack.

        Zeros out ``num_mask_aia_channels`` randomly chosen channels uniformly across all
        timesteps, and independently drops the last channel (HMI) with probability
        ``drop_hmi_probability``.

        Args:
            input_tensor: Array of shape (C, T, H, W).

        Returns:
            Masked array of the same shape.
        """
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
        time_delta_input_minutes: List of candidate input time offsets in minutes, sorted ascending.
            The **last entry must be 0** (the "now" frame, always included). Earlier entries are the
            pool of historical offsets from which ``n_input_timestamps - 1`` frames are sampled
            randomly per call to ``__getitem__``.
        time_delta_target_minutes: Step size in minutes between forecast frames.
            Forecast targets are at offsets 1×, 2×, … (``rollout_steps + 1``)× this value.
            Only used when ``load_forecast_frames=True``.
        n_input_timestamps: Number of input frames to include per sample. Must be
            ≤ ``len(time_delta_input_minutes)``.
        rollout_steps: Number of additional forecast steps beyond the first target frame.
            ``rollout_steps=0`` produces one target frame; ``rollout_steps=N`` produces N+1.
            Only used when ``load_forecast_frames=True``.
        scalers: Per-channel normalization statistics produced by
            ``workshop_infrastructure/utils.py:build_scalers()``. Each entry must expose
            ``.mean``, ``.std``, ``.epsilon``, and ``.sl_scale_factor``.
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
        s3_cache_dir: **Required when reading from S3.** Local directory where S3 files are cached.
            There is no default — you must set this explicitly. Each full-resolution SDO NetCDF file
            is approximately 1 GB, so make sure the target filesystem has sufficient space
            (budget ~1 GB × number of unique timesteps in your dataset).
        s3fs_kwargs: Additional kwargs passed to ``s3fs.S3FileSystem``.
        s3_download_to_temp: If True (recommended for NetCDF/HDF5), download each S3 object to a
            local file before opening. Avoids seekability issues with streaming reads.
        s3_temp_dir: Directory for downloaded S3 files. Defaults to ``s3_cache_dir``.
        s3_boto3_max_concurrency: Number of parallel threads for boto3 multipart downloads.
        s3_boto3_part_size_mb: Part size in MB for boto3 multipart downloads.
        load_forecast_frames: If True (default), load both input and forecast frames and include
            ``forecast`` and ``lead_time_delta`` in the returned sample. Set to False for downstream
            tasks that supply their own target labels (e.g., flare intensity from a separate catalog),
            so that future frames are never fetched from disk or S3. This also relaxes the index
            validity check — only input-frame timestamps need to be present.
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
        s3_cache_dir: str | None = None,
        s3fs_kwargs: dict | None = None,
        s3_download_to_temp: bool = True,
        s3_temp_dir: str | None = None,
        s3_boto3_max_concurrency: int = 4,
        s3_boto3_part_size_mb: int = 64,
        load_forecast_frames: bool = True,
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
        # Note: s3_cache_dir is intentionally left as None here. Its presence is validated
        # lazily in _load_s3_nc_data, so users with only local paths pay no cost.
        self.s3_boto3_max_concurrency = s3_boto3_max_concurrency
        self.s3_boto3_part_size_mb = s3_boto3_part_size_mb
        self._s3fs = None  # lazily initialized per process
        self.load_forecast_frames = load_forecast_frames

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
        # range(1, rollout_steps + 2): starts at 1× (first target), ends at (rollout_steps + 1)×.
        # rollout_steps=0 → one target frame; rollout_steps=N → N+1 target frames.
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

        # Pre-compute normalization arrays once (avoids repeated dict lookups per sample).
        self._means = np.array([self.scalers[ch].mean for ch in self.channels])
        self._stds = np.array([self.scalers[ch].std for ch in self.channels])
        self._epsilons = np.array([self.scalers[ch].epsilon for ch in self.channels])
        self._sl_scale_factors = np.array([self.scalers[ch].sl_scale_factor for ch in self.channels])

    # ------------------------------------------------------------------
    # Index filtering
    # ------------------------------------------------------------------

    def _filter_valid_indices(self) -> list:
        """Return the list of reference timesteps for which all required offsets are present.

        When ``load_forecast_frames=False``, only input-frame offsets are checked, so the
        index does not need to contain future timestamps.
        """
        if self.load_forecast_frames:
            # `+` concatenates two Python lists of np.timedelta64 objects before deduplication.
            time_deltas = np.unique(self.time_delta_input_minutes + self.time_delta_target_minutes)
        else:
            time_deltas = np.unique(self.time_delta_input_minutes)
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

    def __len__(self) -> int:
        """Return the number of valid reference timesteps in this dataset."""
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Load and return a single sample.

        Args:
            idx: Dataset index.

        Returns:
            Dictionary with keys:
                ts (np.ndarray):               (C, T, H, W) — input frames
                time_delta_input (np.ndarray): (T,) — input time offsets in hours (relative to "now")
            When ``load_forecast_frames=True`` (default), also includes:
                forecast (np.ndarray):         (C, L, H, W) — target frames
                lead_time_delta (np.ndarray):  (L,) — lead times in hours (negative = future)
            When ``use_latitude_in_learned_flow=True``, also includes:
                input_latitudes (list[float])
                forecast_latitude (list[float])  — only when ``load_forecast_frames=True``
        """
        self._ensure_logger()
        self.logger.info(f"Retrieving index {idx}.")
        return self._get_index_data(idx)

    # ------------------------------------------------------------------
    # Internal data loading
    # ------------------------------------------------------------------

    def _load_and_stack_frames(self, timesteps) -> np.ndarray:
        """Load, normalize, and stack NetCDF frames into a (C, T, H, W) array.

        Args:
            timesteps: Sequence of timestamps; one NetCDF file is loaded per entry.

        Returns:
            NumPy array of shape (C, T, H, W).
        """
        return np.stack(
            [self.transform_data(self.load_nc_data(self.index.loc[ts, "path"], ts, self.channels))
             for ts in timesteps],
            axis=1,
        )

    def _get_index_data(self, idx: int) -> dict:
        """Build and return the sample dict for a single reference timestep.

        Samples historical input frames randomly from ``time_delta_input_minutes``,
        applies channel masking and optional vertical flip, then assembles the sample
        dict described in ``__getitem__``.
        """
        reference_timestep = self.valid_indices[idx]

        # The last entry in time_delta_input_minutes is always the "now" frame (offset 0
        # relative to the reference timestep). The remaining n-1 input frames are sampled
        # randomly from the earlier offsets, giving the model varied historical context
        # across training steps.
        input_time_deltas = np.array(
            sorted(random.sample(self.time_delta_input_minutes[:-1], self.n_input_timestamps - 1))
            + [self.time_delta_input_minutes[-1]]
        )
        input_timesteps = reference_timestep + input_time_deltas
        stacked_inputs = self._load_and_stack_frames(input_timesteps)

        if self.num_mask_aia_channels > 0 or self.drop_hmi_probability:
            stacked_inputs = self.masker(stacked_inputs)

        # All time deltas are expressed relative to "now" (the last input frame), in hours.
        now_delta = input_time_deltas[-1]
        time_delta_input_float = (
            (now_delta - input_time_deltas) / np.timedelta64(1, "h")
        ).astype(np.float32)

        sample = {
            "ts": stacked_inputs,
            "time_delta_input": time_delta_input_float,
        }

        if self.load_forecast_frames:
            target_time_deltas = np.array(self.time_delta_target_minutes)
            target_timesteps = reference_timestep + target_time_deltas
            sample["forecast"] = self._load_and_stack_frames(target_timesteps)
            sample["lead_time_delta"] = (
                (now_delta - target_time_deltas) / np.timedelta64(1, "h")
            ).astype(np.float32)

        if self.random_vert_flip and torch.bernoulli(torch.ones(()) / 2) == 1:
            sample["ts"] = np.flip(sample["ts"], axis=-2).copy()
            if self.load_forecast_frames:
                sample["forecast"] = np.flip(sample["forecast"], axis=-2).copy()

        if self.use_latitude_in_learned_flow:
            from sunpy.coordinates.ephemeris import get_earth

            sample["input_latitudes"] = [get_earth(ts).lat.value for ts in input_timesteps]
            if self.load_forecast_frames:
                sample["forecast_latitude"] = [get_earth(ts).lat.value for ts in target_timesteps]

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
        self._ensure_logger()
        self.logger.info(f"Reading file {filepath}.")

        if not self._is_s3_path(filepath) and self.sdo_data_root_path and not os.path.isabs(filepath):
            filepath = os.path.join(self.sdo_data_root_path, filepath)

        if self._is_s3_path(filepath):
            return self._load_s3_nc_data(filepath, channels)

        with xr.open_dataset(filepath, engine="h5netcdf", chunks=None, cache=False) as ds:
            return ds[channels].to_array().load().to_numpy()

    def _load_s3_nc_data(self, s3_uri: str, channels: list[str]) -> np.ndarray:
        """Load a NetCDF file from S3 and return the requested channels as a NumPy array.

        Two code paths depending on ``s3_download_to_temp`` (recommended default: True):

        1. **Download-to-cache** (``s3_download_to_temp=True``): the full S3 object is
           downloaded to ``s3_cache_dir`` before opening with xarray. Uses an atomic
           write (partial → rename) so a crashed download never leaves a corrupt cache
           file. Subsequent calls for the same URI are served from the local cache.
           Requires boto3 (preferred, parallel multipart) or s3fs (fallback streaming).

        2. **Streaming** (``s3_download_to_temp=False``): the file is opened in-place
           via fsspec simplecache or direct s3fs. Avoid for NetCDF/HDF5 — these formats
           require random seeks that streaming reads cannot always satisfy.
        """
        if boto3 is None and fsspec is None:
            raise ImportError(
                "S3 support requires either 'boto3' or 'fsspec'+'s3fs'. "
                "Install via: pip install boto3  or  pip install s3fs fsspec"
            )

        if self.s3_cache_dir is None:
            raise ValueError(
                "s3_cache_dir must be set when reading data from S3. "
                "Each full-resolution SDO NetCDF file is approximately 1 GB, so choose a "
                "filesystem with enough free space (budget ~1 GB × number of unique timesteps). "
                "Example: s3_cache_dir='/scratch/my_project/helio_cache'"
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
        """Return True if ``path`` is an S3 URI (starts with ``s3://``)."""
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
            region = (os.environ.get("AWS_REGION")
                      or os.environ.get("AWS_DEFAULT_REGION")
                      or _detect_ec2_region())
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

    def transform_data(self, data: np.ndarray) -> np.ndarray:
        """Apply spatial pooling (if configured) then signum-log normalization.

        Args:
            data: Raw channel array of shape (C, H, W).

        Returns:
            Normalized array of shape (C, H//pooling, W//pooling). See the module-level
            comment for the signum-log math and the ``inverse_transform_data`` method
            to reverse the normalization.
        """
        assert data.ndim == 3
        if self.pooling > 1:
            data = skimage.measure.block_reduce(data, block_size=(1, self.pooling, self.pooling), func=np.mean)
        return transform(data, self._means, self._stds, self._sl_scale_factors, self._epsilons)

    def inverse_transform_data(self, data: np.ndarray) -> np.ndarray:
        """Invert signum-log normalization on a (C, H, W) array.

        Note: this only inverts the *normalization*, not spatial pooling.
        Pooling (applied in ``transform_data``) is not invertible.
        """
        assert data.ndim == 3
        return fast_inverse_transform(data, self._means, self._stds, self._sl_scale_factors, self._epsilons)
