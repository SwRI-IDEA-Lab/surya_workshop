import numpy as np
import pandas as pd
from typing import Callable, Literal
from workshop_infrastructure.datasets.helio import HelioNetCDFDataset


class FlareDSDataset(HelioNetCDFDataset):
    """
    Template child class of HelioNetCDFDataset showing how to build a downstream dataset.
    Extends the base class with a flare intensity label aligned to the Surya index.

    See ``HelioNetCDFDataset`` for all base class parameters (index_path, time_delta_input_minutes,
    scalers, s3_cache_dir, etc.).

    Additional Args:
        load_forecast_frames: If True, also load future Surya frames from S3/disk and include
            ``forecast`` and ``lead_time_delta`` in the sample. Defaults to False because
            flare forecasting uses its own label (``normalized_intensity``), not Surya's future frames.
            Setting this to False avoids downloading the ~1 GB forecast files entirely.
        return_surya_stack: If True (default), include the Surya image stack in the returned dict.
            Set to False to return only the flare intensity label (useful for label inspection).
        max_number_of_samples: Cap the dataset length at this value. Useful for quick experiments.
        label_transform: Optional callable applied to the ``intensity`` column of the flare index
            to produce the ``normalized_intensity`` label. Signature:
            ``(series: pd.Series) -> pd.Series``.  If ``None``, the raw intensity values are
            used as-is. Define this at the call site (e.g., in ``build_datasets()``) to keep
            normalization logic out of the dataset class.
        ds_flare_index_path: Path to the downstream flare intensity CSV index.
        ds_time_column: Column name in the flare index to use as the event timestamp.
        ds_time_tolerance: Maximum allowed time offset when matching Surya and DS indices
            (e.g., ``"15min"``). Unmatched entries are dropped.
        ds_match_direction: Merge direction passed to ``pd.merge_asof``. Use ``"forward"``
            for causal prediction (predict flares from prior solar state).

    Raises:
        ValueError: If ``ds_flare_index_path`` is not provided, or if no overlap exists
            between the Surya and DS indices within the specified tolerance.
    """

    def __init__(
        self,
        # Base class parameters (forwarded to HelioNetCDFDataset)
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
        s3_storage_options: dict | None = None,
        s3_use_simplecache: bool = False,
        s3_cache_dir: str | None = None,
        s3_download_to_temp: bool = True,
        load_forecast_frames: bool = False,
        # Downstream-specific parameters
        return_surya_stack: bool = True,
        max_number_of_samples: int | None = None,
        label_transform: Callable[[pd.Series], pd.Series] | None = None,
        ds_flare_index_path: str | None = None,
        ds_time_column: str | None = None,
        ds_time_tolerance: str | None = None,
        ds_match_direction: Literal["forward", "backward", "nearest"] = "forward",
    ):
        if ds_match_direction not in ["forward", "backward", "nearest"]:
            raise ValueError("ds_match_direction must be one of 'forward', 'backward', or 'nearest'")

        super().__init__(
            index_path=index_path,
            time_delta_input_minutes=time_delta_input_minutes,
            time_delta_target_minutes=time_delta_target_minutes,
            n_input_timestamps=n_input_timestamps,
            rollout_steps=rollout_steps,
            scalers=scalers,
            num_mask_aia_channels=num_mask_aia_channels,
            drop_hmi_probability=drop_hmi_probability,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            channels=channels,
            phase=phase,
            s3_storage_options=s3_storage_options,
            s3_use_simplecache=s3_use_simplecache,
            s3_cache_dir=s3_cache_dir,
            s3_download_to_temp=s3_download_to_temp,
            load_forecast_frames=load_forecast_frames,
        )

        self.return_surya_stack = return_surya_stack

        # Load ds index and find intersection with Surya index
        if ds_flare_index_path is not None:
            self.ds_index = pd.read_csv(ds_flare_index_path)
        else:
            raise ValueError("ds_flare_index_path must be provided for FlareDSDataset")

        self.ds_index["ds_index"] = pd.to_datetime(
            self.ds_index[ds_time_column]
        ).values.astype("datetime64[ns]")
        self.ds_index.sort_values("ds_index", inplace=True)

        # Apply label transform if provided; otherwise use raw intensity values.
        if label_transform is not None:
            self.ds_index["normalized_intensity"] = label_transform(self.ds_index["intensity"])
        else:
            self.ds_index["normalized_intensity"] = self.ds_index["intensity"]

        # Create Surya valid indices and find closest match to DS index
        self.df_valid_indices = pd.DataFrame(
            {"valid_indices": self.valid_indices}
        ).sort_values("valid_indices")
        self.df_valid_indices = pd.merge_asof(
            self.df_valid_indices,
            self.ds_index,
            right_on="ds_index",
            left_on="valid_indices",
            direction=ds_match_direction,
        )
        # Remove duplicates keeping closest match
        self.df_valid_indices["index_delta"] = np.abs(
            self.df_valid_indices["valid_indices"] - self.df_valid_indices["ds_index"]
        )
        self.df_valid_indices = self.df_valid_indices.sort_values(
            ["ds_index", "index_delta"]
        )
        self.df_valid_indices.drop_duplicates(
            subset="ds_index", keep="first", inplace=True
        )
        # Enforce a maximum time tolerance for matches
        if ds_time_tolerance is not None:
            self.df_valid_indices = self.df_valid_indices.loc[
                self.df_valid_indices["index_delta"] <= pd.Timedelta(ds_time_tolerance),
                :,
            ]
            if len(self.df_valid_indices) == 0:
                raise ValueError("No intersection between Surya and DS indices")

        # Override valid indices variables to reflect matches between Surya and DS
        self.valid_indices = [
            pd.Timestamp(date) for date in self.df_valid_indices["valid_indices"]
        ]
        self.adjusted_length = len(self.valid_indices)
        self.df_valid_indices.set_index("valid_indices", inplace=True)

        if max_number_of_samples is not None:
            self.adjusted_length = min(self.adjusted_length, max_number_of_samples)

    def __len__(self):
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Dataset index.

        Returns:
            Dictionary containing:
                forecast (np.float32): Normalized log10 flare intensity label.
                ds_index (str): ISO-format timestamp from the flare index.
            When ``return_surya_stack=True``, also includes all keys from
            ``HelioNetCDFDataset.__getitem__`` (ts, time_delta_input, lead_time_delta, etc.).
        """
        sample = super().__getitem__(idx=idx) if self.return_surya_stack else {}
        sample["forecast"] = self.df_valid_indices.iloc[idx]["normalized_intensity"].astype(np.float32)
        sample["ds_index"] = self.df_valid_indices["ds_index"].iloc[idx].isoformat()
        return sample
