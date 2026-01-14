import os
import sys
import random
import tempfile
from datetime import datetime
from typing import Tuple
import torch
import numpy as np
import skimage.measure
import xarray as xr
import pandas as pd
from logging import Logger
from torch.utils.data import Dataset
from Surya.surya.utils.distributed import get_rank
from Surya.surya.utils.log import create_logger
from functools import cache

# Optional S3 support (required only when reading s3:// URIs)
try:
    import fsspec
    import s3fs
except Exception:  # pragma: no cover
    fsspec = None
    s3fs = None

# Optional boto3 support (faster whole-object downloads from S3)
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


@njit(parallel=True)
def fast_transform(data, means, stds, sl_scale_factors, epsilons):
    """
    Implements signum log transform using numba for speed
    Notes:
    - This must reside outside the class definition from which it is called.
    - We used this function during pretraining for faster data loading. On select
      GPU clusters it leads to the system hanging however when data loading happens
      outside the GPU thread. See below for a non-numba-enhanced version.

    Args:
        data: Numpy array of shape C, H, W
        means: Numpy array of shape C. Mean per channel.
        stds: Numpy array of shape C. Standard deviation per channel.
        sl_scale_factors: Numpy array of shape C. Signum-log scale factors.
        epsilons: Numpy array of shape C. Constant to avoid zero division.

    Returns:
        Numpy array of shape C, H, W.
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
                val = data[c, i, j]
                val = val * sl_scale_factor
                if val >= 0:
                    val = np.log1p(val)
                else:
                    val = -np.log1p(-val)
                out[c, i, j] = (val - mean) / (std + eps)
    return out

@njit(parallel=True)
def fast_inverse_transform(data, means, stds, sl_scale_factors, epsilons):
    """
    implements signum log transform using numba for speed
        note: this must reside outside the class definition from which it is called
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
                val = data[c, i, j]
                val = val * (std + eps) + mean

                # Apply the inverse of log1p, which is expm1
                abs_data = np.expm1(np.abs(val))

                # Restore the original sign
                out[c, i, j] = np.sign(val) * abs_data / sl_scale_factor
    return out


def transform(
    data: np.ndarray, means: np.ndarray, stds: np.ndarray, sl_scale_factors: np.ndarray, epsilons: np.ndarray
) -> np.ndarray:
    """
    Implements signum log transform. Drop-in replacement for
    `fast_transform` method above.

    Args:
        data: Numpy array of shape C, H, W
        means: Numpy array of shape C. Mean per channel.
        stds: Numpy array of shape C. Standard deviation per channel.
        sl_scale_factors: Numpy array of shape C. Signum-log scale factors.
        epsilons: Numpy array of shape C. Constant to avoid zero division.

    Returns:
        Numpy array of shape C, H, W.
    """
    means = means.reshape(*means.shape, 1, 1)
    stds = stds.reshape(*stds.shape, 1, 1)
    sl_scale_factors = sl_scale_factors.reshape(*sl_scale_factors.shape, 1, 1)
    epsilons = epsilons.reshape(*epsilons.shape, 1, 1)

    data = data * sl_scale_factors
    data = np.sign(data) * np.log1p(np.abs(data))
    data = (data - means) / (stds + epsilons)

    return data


@njit(parallel=True)
def inverse_fast_transform(data, means, stds, sl_scale_factors, epsilons):
    """
    Implements inverse signum log transform using numba for speed

    Args:
        data: Numpy array of shape C, H, W
        means: Numpy array of shape C. Mean per channel.
        stds: Numpy array of shape C. Standard deviation per channel.
        sl_scale_factors: Numpy array of shape C. Signum-log scale factors.
        epsilons: Numpy array of shape C. Constant to avoid zero division.

    Returns:
        Numpy array of shape C, H, W.
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
                val = data[c, i, j]
                val = val * (std + eps) + mean

                if val >= 0:
                    val = np.expm1(val)
                else:
                    val = -np.expm1(-val)

                val = val / sl_scale_factor

                out[c, i, j] = val

    return out


def inverse_transform_single_channel(data, mean, std, sl_scale_factor, epsilon):
    """
    Implements inverse signum log transform.

    Args:
        data: Numpy array of shape C, H, W
        means: Numpy array of shape C. Mean per channel.
        stds: Numpy array of shape C. Standard deviation per channel.
        sl_scale_factors: Numpy array of shape C. Signum-log scale factors.
        epsilons: Numpy array of shape C. Constant to avoid zero division.

    Returns:
        Numpy array of shape C, H, W.
    """
    data = data * (std + epsilon) + mean

    data = np.sign(data) * np.expm1(np.abs(data))

    data = data / sl_scale_factor

    return data


class RandomChannelMaskerTransform:
    def __init__(self, num_channels, num_mask_aia_channels, phase, drop_hmi_probability):
        """
        Initialize the RandomChannelMaskerTransform class as a transform.

        Args:
        - num_channels: Total number of channels in the input (3rd dimension of
          the tensor).
        - num_mask_aia_channels: Number of channels to randomly mask.
        """
        self.num_channels = num_channels
        self.num_mask_aia_channels = num_mask_aia_channels
        self.drop_hmi_probability = drop_hmi_probability

    def __call__(self, input_tensor):
        C, T, H, W = input_tensor.shape  # Unpacking the correct 5 dimensions

        # Randomly select channels to mask
        channels_to_mask = random.sample(range(C), self.num_mask_aia_channels)

        # Create an in-place mask of shape [1, 1, num_channels, 1, 1]
        mask = torch.ones((C, 1, 1, 1))
        mask[channels_to_mask, ...] = 0  # Set selected channels to zero

        # Apply the mask in-place for memory efficiency
        masked_tensor = input_tensor * mask  # Modify input_tensor directly

        if self.drop_hmi_probability > random.random():
            masked_tensor[-1, ...] = 0

        return masked_tensor


class HelioNetCDFDataset(Dataset):
    """
    PyTorch dataset to load a curated dataset from the NASA Solar Dynamics
    Observatory (SDO) mission stored as NetCDF files, with handling for variable timesteps.

    Internally maintains two databases. The first is `self.index`. This takes the
    form
                                                                        path  present
        timestep
        2011-01-01 00:00:00  /lustre/fs0/scratch/shared/data/2011/01/Arka_2...        1
        2011-01-01 00:12:00  /lustre/fs0/scratch/shared/data/2011/01/Arka_2...        1
        ...                                                                ...      ...
        2012-11-30 23:48:00  /lustre/fs0/scratch/shared/data/2012/11/Arka_2...        1

    The second is `self.valid_indices`. This is simply a list of timesteps -- entries
    in the index of `self.index` -- which define valid samples. A sample is valid
    when all timestamps that can be reached by entris in
    time_delta_input_minutes and time_delta_target_minutes can be reached from it
    are present.
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
        use_latitude_in_learned_flow=False,
        channels: list[str] | None = None,
        phase="train",
        pooling: int | None = None,
        random_vert_flip: bool = False,
        sdo_data_root_path: str | None = None,
        # --- S3 options (used only if index contains s3:// URIs) ---
        s3_storage_options: dict | None = None,
        s3_use_simplecache: bool = False,
        s3_cache_dir: str = "/tmp/helio_s3_cache",
        s3fs_kwargs: dict | None = None,
        # --- Ephemeral download-to-local-then-open (recommended for NetCDF/HDF5 on S3) ---
        # If True, S3 objects are downloaded to a temporary local file, opened locally
        # with xarray/h5netcdf, and deleted immediately after loading into memory.
        s3_download_to_temp: bool = True,
        s3_temp_dir: str | None = "/tmp/helio_s3_cache",
        # boto3 transfer tuning (used only when boto3 is available; otherwise falls back to s3fs/fsspec)
        s3_boto3_max_concurrency: int = 2,
        s3_boto3_part_size_mb: int = 64,
    ):
        self.scalers = scalers
        self.phase = phase
        self.channels = channels
        self.num_mask_aia_channels = num_mask_aia_channels
        self.drop_hmi_probability = drop_hmi_probability
        self.n_input_timestamps = n_input_timestamps
        self.rollout_steps = rollout_steps
        self.use_latitude_in_learned_flow = use_latitude_in_learned_flow
        self.pooling = pooling if pooling is not None else 1
        self.random_vert_flip = random_vert_flip
        self.sdo_data_root_path = sdo_data_root_path
        # --- S3 configuration ---
        # These are only used when a filepath begins with 's3://'.
        self.s3_storage_options = s3_storage_options or {}
        self.s3_use_simplecache = s3_use_simplecache
        self.s3_cache_dir = s3_cache_dir
        self.s3fs_kwargs = s3fs_kwargs or {}
        self.s3_download_to_temp = s3_download_to_temp
        self.s3_temp_dir = s3_temp_dir
        self.s3_boto3_max_concurrency = s3_boto3_max_concurrency
        self.s3_boto3_part_size_mb = s3_boto3_part_size_mb
        # Lazily-initialized S3 filesystem handle (per-process).
        self._s3fs = None

        if self.channels is None:
            # AIA + HMI channels
            self.channels = [
                "0094",
                "0131",
                "0171",
                "0193",
                "0211",
                "0304",
                "0335",
                "hmi",
            ]
        self.in_channels = len(self.channels)

        self.masker = RandomChannelMaskerTransform(
            num_channels=self.in_channels,
            num_mask_aia_channels=self.num_mask_aia_channels,
            phase=self.phase,
            drop_hmi_probability=self.drop_hmi_probability,
        )

        # Convert time delta to numpy timedelta64
        self.time_delta_input_minutes = sorted(np.timedelta64(t, "m") for t in time_delta_input_minutes)
        self.time_delta_target_minutes = [
            np.timedelta64(iroll * time_delta_target_minutes, "m") for iroll in range(1, rollout_steps + 2)
        ]

        # Create the index
        self.index = pd.read_csv(index_path)
        self.index = self.index[self.index["present"] == 1]
        self.index["timestep"] = pd.to_datetime(self.index["timestep"]).values.astype("datetime64[ns]")
        self.index.set_index("timestep", inplace=True)
        self.index.sort_index(inplace=True)

        # Filter out rows where the sequence is not fully present
        self.valid_indices = self.filter_valid_indices()
        self.adjusted_length = len(self.valid_indices)

        self.rank = get_rank()
        self.logger: Logger | None = None

    def create_logger(self):
        """
        Creates a logger attached to self.logger.
        The logger is identified by SLURM job ID
        as well as the data processes rank and process ID.
        """
        os.makedirs("logs/data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        pid = os.getpid()
        self.logger = create_logger(
            output_dir="logs/data",
            dist_rank=self.rank,
            name=f"{timestamp}_{self.rank:>03}_data_{self.phase}_{pid}",
        )

    def filter_valid_indices(self):
        """
        Extracts timestamps from the index of self.index that define valid
        samples.

        Args:
        Returns:
            List of timestamps.
        """

        valid_indices = []
        time_deltas = np.unique(self.time_delta_input_minutes + self.time_delta_target_minutes)

        for reference_timestep in self.index.index:
            required_timesteps = reference_timestep + time_deltas

            if all(t in self.index.index for t in required_timesteps):
                valid_indices.append(reference_timestep)

        return valid_indices

    def __len__(self):
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Index of sample to load. (Pytorch standard.)
        Returns:
            Dictionary with following keys. The values are tensors with shape as follows:
                ts (torch.Tensor):                C, T, H, W
                time_delta_input (torch.Tensor):  T
                input_latitude (torch.Tensor):    T
                forecast (torch.Tensor):          C, L, H, W
                lead_time_delta (torch.Tensor):   L
                forecast_latitude (torch.Tensor): L
            C - Channels, T - Input times, H - Image height, W - Image width, L - Lead time.
        """
        if self.logger is None:
            self.create_logger()
            self.logger.info(f"HelioNetCDFDataset of length {self.__len__()}.")

        exception_counter = 0
        max_exception = 100

        self.logger.info(f"Starting to retrieve index {idx}.")

        while True:
            try:
                sample = self._get_index_data(idx)
            except Exception as e:
                exception_counter += 1
                if exception_counter >= max_exception:
                    raise e

                reference_timestep = self.valid_indices[idx]
                self.logger.warning(
                    f"Failed retrieving index {idx}. Timestamp {reference_timestep}. Attempt {exception_counter}."
                )

                idx = (idx + 1) % self.__len__()
            else:
                self.logger.info(f"Returning index {idx}.")
                return sample

    def _get_index_data(self, idx: int) -> dict:
        """
        Args:
            idx: Index of sample to load. (Pytorch standard.)
        Returns:
            Dictionary with following keys. The values are tensors with shape as follows:
                ts (torch.Tensor):                C, T, H, W
                time_delta_input (torch.Tensor):  T
                input_latitude (torch.Tensor):    T
                forecast (torch.Tensor):          C, L, H, W
                lead_time_delta (torch.Tensor):   L
                forecast_latitude (torch.Tensor): L
            C - Channels, T - Input times, H - Image height, W - Image width, L - Lead time.
        """
        # start_time = time.time()

        time_deltas = np.array(
            sorted(random.sample(self.time_delta_input_minutes[:-1], self.n_input_timestamps - 1))
            + [self.time_delta_input_minutes[-1]]
            + self.time_delta_target_minutes
        )
        reference_timestep = self.valid_indices[idx]
        required_timesteps = reference_timestep + time_deltas

        sequence_data = [
            self.transform_data(self.load_nc_data(self.index.loc[timestep, "path"], timestep, self.channels))
            for timestep in required_timesteps
        ]

        # Split sequence_data into inputs and target
        inputs = sequence_data[: -self.rollout_steps - 1]
        targets = sequence_data[-self.rollout_steps - 1 :]

        stacked_inputs = np.stack(inputs, axis=1)
        stacked_targets = np.stack(targets, axis=1)

        timestamps_input = required_timesteps[: -self.rollout_steps - 1]
        timestamps_targets = required_timesteps[-self.rollout_steps - 1 :]

        if self.num_mask_aia_channels > 0 or self.drop_hmi_probability:
            # assert 0 < self.num_mask_aia_channels < self.in_channels, \
            #     f'num_mask_aia_channels = {self.num_mask_aia_channels} should lie between 0 and {self.in_channels}'

            stacked_inputs = self.masker(stacked_inputs)

        time_delta_input_float = (
            time_deltas[-self.rollout_steps - 2] - time_deltas[: -self.rollout_steps - 1]
        ) / np.timedelta64(1, "h")
        time_delta_input_float = time_delta_input_float.astype(np.float32)

        lead_time_delta_float = (
            time_deltas[-self.rollout_steps - 2] - time_deltas[-self.rollout_steps - 1 :]
        ) / np.timedelta64(1, "h")
        lead_time_delta_float = lead_time_delta_float.astype(np.float32)

        # print('LocalRank', int(os.environ["LOCAL_RANK"]),
        #       'GlobalRank', int(os.environ["RANK"]),
        #       'worker', torch.utils.data.get_worker_info().id,
        #       f': Processed Input: {idx} ',time.time()- start_time)

        metadata = {
            "timestamps_input": timestamps_input,
            "timestamps_targets": timestamps_targets,
        }

        if self.random_vert_flip:
            if torch.bernoulli(torch.ones(()) / 2) == 1:
                stacked_inputs = torch.flip(stacked_inputs, dims=-2)
                stacked_targets = torch.flip(stacked_inputs, dims=-2)

        if self.use_latitude_in_learned_flow:
            from sunpy.coordinates.ephemeris import get_earth

            sequence_latitude = [get_earth(timestep).lat.value for timestep in required_timesteps]
            input_latitudes = sequence_latitude[: -self.rollout_steps - 1]
            target_latitude = sequence_latitude[-self.rollout_steps - 1 :]

            return {
                "ts": stacked_inputs,
                "time_delta_input": time_delta_input_float,
                "input_latitudes": input_latitudes,
                "forecast": stacked_targets,
                "lead_time_delta": lead_time_delta_float,
                "forecast_latitude": target_latitude,
            }  # , metadata

        return {
            "ts": stacked_inputs,
            "time_delta_input": time_delta_input_float,
            "forecast": stacked_targets,
            "lead_time_delta": lead_time_delta_float,
        }  # , metadata

    def _is_s3_path(self, path: str) -> bool:
        """Return True if `path` is an S3 URI (s3://...)."""
        return isinstance(path, str) and path.startswith("s3://")

    def _get_s3fs(self):
        """Lazily create and return an S3FileSystem instance.

        Notes
        -----
        - Requires `s3fs` to be installed (imported at module import time).
        - Uses the standard AWS credential provider chain by default (e.g.,
          instance profile / environment variables).
        """
        if s3fs is None:
            raise ImportError(
                "S3 support requires the 's3fs' and 'fsspec' packages. " "Install them (e.g., pip install s3fs fsspec)."
            )
        if self._s3fs is None:
            self._s3fs = s3fs.S3FileSystem(**self.s3fs_kwargs, **self.s3_storage_options)

        return self._s3fs

    def _parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        """Parse s3://bucket/key into (bucket, key)."""
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Not an S3 URI: {s3_uri}")
        path = s3_uri[5:]
        bucket, key = path.split("/", 1)
        return bucket, key

    def _download_s3_to_path(self, s3_uri: str, local_path: str) -> None:
        """Download an S3 object to `local_path`.

        Uses boto3's transfer manager when available (faster, parallel multipart),
        otherwise falls back to streaming via s3fs.
        """
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

        # Prefer boto3 for high-throughput whole-object downloads (esp. public buckets).
        if boto3 is not None and TransferConfig is not None:
            anon = bool(self.s3_storage_options.get("anon") or self.s3fs_kwargs.get("anon"))
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")

            if anon:
                client = boto3.client(
                    "s3",
                    region_name=region,
                    config=BotoConfig(
                        signature_version=UNSIGNED,
                        max_pool_connections=max(32, int(self.s3_boto3_max_concurrency) * 2),
                        retries={"max_attempts": 10, "mode": "adaptive"},
                    ),
                )
            else:
                client = boto3.client(
                    "s3",
                    region_name=region,
                    config=BotoConfig(
                        max_pool_connections=max(32, int(self.s3_boto3_max_concurrency) * 2),
                        retries={"max_attempts": 10, "mode": "adaptive"},
                    ),
                )

            tcfg = TransferConfig(
                multipart_threshold=int(self.s3_boto3_part_size_mb) * 1024 * 1024,
                multipart_chunksize=int(self.s3_boto3_part_size_mb) * 1024 * 1024,
                max_concurrency=int(self.s3_boto3_max_concurrency),
                use_threads=True,
                io_chunksize=1024 * 1024,
            )

            bucket, key = self._parse_s3_uri(s3_uri)
            client.download_file(bucket, key, local_path, Config=tcfg)
            return

        # Fallback: use s3fs to stream the object to disk.
        fs = self._get_s3fs()
        with fs.open(s3_uri, "rb") as src, open(local_path, "wb") as dst:
            for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                dst.write(chunk)

    def load_nc_data(self, filepath: str, timestep: pd.Timestamp, channels: list[str]) -> np.ndarray:
        """
        Load a NetCDF file from either local disk or S3 and return channel-stacked data.

        Parameters
        ----------
        filepath : str
            Path to NetCDF file. May be a local filesystem path or an S3 URI
            (e.g., s3://bucket/key).
        timestep : pandas.Timestamp
            Timestamp associated with the sample (used for logging/debugging).
        channels : list[str]
            Variable names to extract from the dataset, stacked into (C, H, W).

        Returns
        -------
        numpy.ndarray
            Array of shape (C, H, W).

        Notes
        -----
        Performance recommendations (AWS):
        - Prefer running compute in the same AWS region as the S3 bucket.
        - Use IAM instance roles / task roles rather than static credentials.
        - Enable `s3_use_simplecache=True` to cache objects locally and reduce
          repeated S3 range reads. Point `s3_cache_dir` to fast ephemeral storage
          when available (e.g., NVMe instance store).

        Implementation detail:
        - When using `simplecache::`, fsspec expects `target_protocol` and
          `target_options` for the chained filesystem. Do not pass an S3FileSystem
          instance in kwargs (e.g., {"s3": fs}), as that will raise a TypeError.
        """
        self.logger.info(f"Reading file {filepath}.")

        # If a relative/local path and a root is provided, make it absolute.
        # Do NOT rewrite S3 URIs.
        if (not self._is_s3_path(filepath)) and self.sdo_data_root_path and not os.path.isabs(filepath):
            filepath = os.path.join(self.sdo_data_root_path, filepath)

        # ---- S3 path handling ----
        if self._is_s3_path(filepath):
            if fsspec is None and boto3 is None:
                raise ImportError(
                    "S3 support requires either 'boto3' or ('s3fs' and 'fsspec'). "
                    "Install them (e.g., pip install boto3) or (pip install s3fs fsspec)."
                )

            # Recommended for NetCDF/HDF5: download whole object to a local temp file,
            # open locally (avoids many small S3 range reads), then delete immediately
            # after data is loaded into memory.
            if self.s3_download_to_temp:
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".nc",
                        delete=False,
                        dir=self.s3_temp_dir,
                    ) as tmp:
                        tmp_path = tmp.name

                    self._download_s3_to_path(filepath, tmp_path)

                    with xr.open_dataset(tmp_path, engine="h5netcdf", chunks=None, cache=False) as ds:
                        data = ds[channels].to_array().load().to_numpy()
                    return data
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

            # If you explicitly want read-through caching rather than ephemeral temp files,
            # keep the existing fsspec/s3fs path.
            if fsspec is None:
                raise ImportError(
                    "Read-through caching requires 'fsspec' (and typically 's3fs'). "
                    "Install them (e.g., pip install s3fs fsspec)."
                )

            # Optional local caching (read-through).
            # For chained FS (simplecache), use target_protocol/target_options.
            if self.s3_use_simplecache:
                cached_url = f"simplecache::{filepath}"

                # Options for the underlying S3 filesystem must be a *dict* (not an S3FileSystem object)
                s3_options = {**self.s3_storage_options, **self.s3fs_kwargs}

                opener = fsspec.open(
                    cached_url,
                    mode="rb",
                    cache_storage=self.s3_cache_dir,
                    s3=s3_options,
                )
            else:
                fs = self._get_s3fs()
                opener = fs.open(filepath, mode="rb")

            with opener as f:
                # h5netcdf/h5py require a seekable file-like object; s3fs provides this
                # via ranged GET requests.
                with xr.open_dataset(f, engine="h5netcdf", chunks=None, cache=False) as ds:
                    data = ds[channels].to_array().load().to_numpy()

            return data

        # ---- Local path handling ----
        with xr.open_dataset(filepath, engine="h5netcdf", chunks=None, cache=False) as ds:
            data = ds[channels].to_array().load().to_numpy()

        return data

    @cache
    def transformation_inputs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        means = np.array([self.scalers[ch].mean for ch in self.channels])
        stds = np.array([self.scalers[ch].std for ch in self.channels])
        epsilons = np.array([self.scalers[ch].epsilon for ch in self.channels])
        sl_scale_factors = np.array([self.scalers[ch].sl_scale_factor for ch in self.channels])

        return means, stds, epsilons, sl_scale_factors
    
    def transform_data(self, data: np.ndarray) -> np.ndarray:
        """
        Applies scalers.

        Args:
            data: Numpy array of shape (C, H, W)
        Returns:
            Tensor of shape (C, H, W). Data type float32.
        Uses:
                numba to speed up transform
                tvk-srm-heliofm  environment cloned from srm-heliofm with numba added
                tvk_dgx_slurm.sh  shell script modified to use new environment and new jobname
                train_spectformer_dgx.yaml new jobname
        """
        assert data.ndim == 3

        if self.pooling > 1:
            data = skimage.measure.block_reduce(data, block_size=(1, self.pooling, self.pooling), func=np.mean)

        means, stds, epsilons, sl_scale_factors = self.transformation_inputs()
        result_np = transform(data, means, stds, sl_scale_factors, epsilons)
        return result_np    

    def inverse_transform_data(self, data: np.ndarray) -> np.ndarray:
        """
        Applies scalers.

        Args:
            data: Numpy array of shape (C, H, W)
        Returns:
            Tensor of shape (C, H, W). Data type float32.
        Uses:
                numba to speed up transform
                tvk-srm-heliofm  environment cloned from srm-heliofm with numba added
                tvk_dgx_slurm.sh  shell script modified to use new environment and new jobname
                train_spectformer_dgx.yaml new jobname
        """
        assert data.ndim == 3

        if self.pooling > 1:
            data = skimage.measure.block_reduce(
                data,
                block_size=(1, self.pooling, self.pooling),
                func=np.mean
            )

        means, stds, epsilons, sl_scale_factors = self.transformation_inputs()
        result_np = fast_inverse_transform(data, means, stds, sl_scale_factors, epsilons)
        return result_np