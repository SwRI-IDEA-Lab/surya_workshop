import numpy as np
import pandas as pd
from typing import Callable, Literal
from workshop_infrastructure.datasets.helio import HelioNetCDFDataset

from downstream_apps.sep_pred_psp.datasets.event_sampling import (
    load_event_list,
    match_catalog_to_surya,
    sample_balanced,
)


class SepPspDSDataset(HelioNetCDFDataset):
    """
    Child class of HelioNetCDFDataset for SEP prediction from PSP-aligned SHARP data.
    Extends the base class with a Jlinlin label aligned to the Surya index, sampled half
    from inside PSP/ISOIS SEP events and half from quiet times (see ``event_sampling.py``).

    All ``HelioNetCDFDataset`` keyword arguments (``index_path``, ``scalers``, ``channels``,
    ``s3_cache_dir``, etc.) are accepted via ``**kwargs`` and forwarded to the base class.
    ``load_forecast_frames`` defaults to ``False`` here (this task supplies its own labels,
    so future Surya frames are never fetched); pass it explicitly to override.

    Additional Args:
        return_surya_stack: If True (default), include the Surya image stack in the returned dict.
            Set to False to return only the Jlinlin label (useful for label inspection).
        max_number_of_samples: Total samples to draw: N//2 events + N - N//2 non-events.
            ``None`` uses every available event plus the same number of non-events.
            Pass this per split (``train_kwargs``/``val_kwargs`` of ``build_helio_datasets``)
            to size the training set independently of the validation set.
        max_frames_per_event: Ceiling on how many Surya frames one SEP event may contribute.
            1 (the default) means one sample per event, so the split can hold no more event
            samples than it has events. Higher values let the draw take further frames from
            each event window, spread as far apart in PSP time as the window allows.
        label_column: Column of the SEP/PSP index used as the label. ``Jlinlin`` is the
            smoothed series (~0.02 dex hour to hour); ``Jlinlin_raw`` is the unsmoothed one
            (~0.12 dex), which carries no processing of unknown provenance but is noisier.
        label_transform: Optional callable applied to ``label_column`` to produce the
            ``normalized_intensity`` label. Signature: ``(series: pd.Series) -> pd.Series``.
            If ``None``, the column is used as-is -- note that raw Jlinlin spans ~6 decades,
            so MSE on it is dominated by a handful of samples; see ``label_transform.py``.
            Define this at the call site (e.g., in ``build_datasets()``) to keep
            normalization logic out of the dataset class, and share one instance across
            splits so they are scored on the same scale.
        ds_sep_psp_index_path: Path to the downstream SHARP-to-PSP CSV index.
        ds_event_list_path: Path to the PSP/ISOIS SEP event list CSV.
        ds_time_column: Column name in the SEP/PSP index used to pick the Surya frame.
        ds_time_tolerance: Maximum allowed time offset when matching Surya and DS indices
            (e.g., ``"15min"``). Unmatched entries are dropped.
        ds_match_direction: ``"forward"`` uses the Surya frame at or before the catalog time
            (causal prediction); ``"backward"`` the one after; ``"nearest"`` either.
        ds_non_event_buffer: Minimum gap (PSP time) between a non-event sample and any event
            window, e.g. ``"1d"``.
        sample_seed: Seed for the random event / non-event draw.

    Raises:
        ValueError: If a required path is not provided, or if no catalog row can be
            matched to the Surya index within the specified tolerance.
    """

    def __init__(
        self,
        # Downstream-specific parameters
        return_surya_stack: bool = True,
        max_number_of_samples: int | None = None,
        max_frames_per_event: int = 1,
        label_column: str = "Jlinlin",
        label_transform: Callable[[pd.Series], pd.Series] | None = None,
        ds_sep_psp_index_path: str | None = None,
        ds_event_list_path: str | None = None,
        ds_time_column: str | None = None,
        ds_time_tolerance: str | None = None,
        ds_match_direction: Literal["forward", "backward", "nearest"] = "forward",
        ds_non_event_buffer: str = "1d",
        sample_seed: int = 0,
        # All HelioNetCDFDataset parameters (index_path, scalers, channels, s3_*, etc.)
        **kwargs,
    ):
        if ds_match_direction not in ["forward", "backward", "nearest"]:
            raise ValueError("ds_match_direction must be one of 'forward', 'backward', or 'nearest'")
        if ds_sep_psp_index_path is None:
            raise ValueError("ds_sep_psp_index_path must be provided for SepPspDSDataset")
        if ds_event_list_path is None:
            raise ValueError("ds_event_list_path must be provided for SepPspDSDataset")

        # load_forecast_frames defaults to False here: this task supplies its own
        # labels, so future Surya frames never need to be fetched from disk/S3.
        kwargs.setdefault("load_forecast_frames", False)
        super().__init__(**kwargs)

        self.return_surya_stack = return_surya_stack

        catalog = pd.read_csv(ds_sep_psp_index_path)
        if label_column not in catalog.columns:
            raise ValueError(
                f"label_column {label_column!r} is not a column of {ds_sep_psp_index_path}. "
                f"Available columns: {sorted(catalog.columns)}"
            )
        # Apply the label transform if provided; otherwise use the column as-is.
        if label_transform is not None:
            catalog["normalized_intensity"] = label_transform(catalog[label_column])
        else:
            catalog["normalized_intensity"] = catalog[label_column]

        # Pair every catalog row with a Surya frame from this split, then draw the balanced
        # event / non-event sample. The chosen rows stay in df_valid_indices for inspection.
        matched = match_catalog_to_surya(
            catalog, self.valid_indices, ds_time_column, ds_time_tolerance, ds_match_direction
        )
        if len(matched) == 0:
            raise ValueError("No intersection between Surya and DS indices")

        self.df_valid_indices = sample_balanced(
            matched,
            load_event_list(ds_event_list_path),
            n_samples=max_number_of_samples,
            seed=sample_seed,
            non_event_buffer=ds_non_event_buffer,
            max_frames_per_event=max_frames_per_event,
        )

        # Override valid indices variables to reflect the selected samples
        self.valid_indices = [pd.Timestamp(date) for date in self.df_valid_indices["valid_indices"]]
        self.adjusted_length = len(self.valid_indices)
        self.df_valid_indices.set_index("valid_indices", inplace=True)

    def __len__(self):
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Dataset index.

        Returns:
            Dictionary containing:
                forecast (np.float32): Jlinlin label (after ``label_transform``, if any).
                is_event (int): 1 if the sample is from inside an SEP event, else 0.
                frame_round (int): which pass over the events this frame came from — 1 for
                    the frame nearest the event's Max Time Lo, 2+ for further frames of an
                    event already sampled, 0 for non-events.
                event_number (str): ISOIS event number, or "" for non-events.
                ds_index (str): ISO-format catalog timestamp used to pick the Surya frame.
                valid_index (str): ISO-format Surya timestep — the sample's identity. Saved
                    prediction files key on this, so a row can be traced back to a frame.
            When ``return_surya_stack=True``, also includes all keys from
            ``HelioNetCDFDataset.__getitem__`` (ts, time_delta_input, lead_time_delta, etc.).
        """
        sample = super().__getitem__(idx=idx) if self.return_surya_stack else {}
        row = self.df_valid_indices.iloc[idx]
        sample["forecast"] = np.float32(row["normalized_intensity"])
        sample["is_event"] = int(row["is_event"])
        sample["event_number"] = row["event_number"]
        sample["frame_round"] = int(row["frame_round"])
        sample["ds_index"] = row["ds_index"].isoformat()
        # row.name is the df's index, i.e. the Surya timestep this label was matched to.
        sample["valid_index"] = row.name.isoformat()
        return sample
