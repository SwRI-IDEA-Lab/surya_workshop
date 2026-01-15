import numpy as np
import pandas as pd
from typing import Literal

from workshop_infrastructure.datasets.helio_aws import HelioNetCDFDatasetAWS


class EVESpectraDSDataset(HelioNetCDFDatasetAWS):
    """
    Downstream dataset for EVE spectra stored in a CSV where:
      - One column is a timestamp (e.g., 'timestamp')
      - All remaining columns are wavelengths (as strings or numbers), each row is a spectrum

    This follows the same pattern as FlareDSDataset in template_dataset.py:
      1) Initialize HelioNetCDFDatasetAWS (parent)
      2) Load downstream index (CSV)
      3) Normalize downstream target
      4) Merge Surya valid indices with downstream timestamps via merge_asof
      5) Override valid_indices to matched subset
      6) __getitem__ returns Surya stack (optional) + "forecast" (1D array) + "ds_index"

    Key difference from template:
      - base_dictionary["forecast"] is a 1D np.ndarray (float32) of length n_wavelengths
        instead of a scalar.
    """

    def __init__(
        self,
        #### Parent HelioNetCDFDatasetAWS parameters (required)
        index_path: str,
        time_delta_input_minutes: list[int],
        time_delta_target_minutes: int,
        n_input_timestamps: int,
        rollout_steps: int,
        scalers=None,
        num_mask_aia_channels=0,
        drop_hmi_probability=0,
        use_latitude_in_learned_flow=False,
        channels: list[str] | None = None,
        phase="train",
        s3_use_simplecache: bool = True,
        s3_cache_dir: str = "/tmp/helio_s3_cache",
        #### Downstream (DS) specific parameters
        return_surya_stack: bool = True,
        max_number_of_samples: int | None = None,
        ds_spectra_csv_path: str | None = None,
        ds_time_column: str = "timestamp",
        ds_time_tolerance: str | None = None,
        ds_match_direction: Literal["forward", "backward", "nearest"] = "forward",
        # Normalization controls (kept simple but robust)
        apply_log10: bool = True,
        normalize_per_wavelength: bool = True,
        eps: float = 1e-30,
    ):
        if ds_match_direction not in ["forward", "backward", "nearest"]:
            raise ValueError("ds_match_direction must be one of 'forward', 'backward', or 'nearest'")

        # 1) Initialize parent
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
            s3_use_simplecache=s3_use_simplecache,
            s3_cache_dir=s3_cache_dir,
        )

        self.return_surya_stack = return_surya_stack

        # 2) Load downstream CSV
        if ds_spectra_csv_path is None:
            raise ValueError("ds_spectra_csv_path must be provided for EVESpectraDSDataset")

        ds = pd.read_csv(ds_spectra_csv_path)

        if ds_time_column not in ds.columns:
            raise ValueError(
                f"ds_time_column='{ds_time_column}' not found in CSV columns. "
                f"Available columns include: {list(ds.columns)[:10]} ..."
            )

        # Parse and sort DS timestamps
        ds["ds_index"] = pd.to_datetime(ds[ds_time_column]).values.astype("datetime64[ns]")
        ds.sort_values("ds_index", inplace=True)

        # Identify wavelength columns (everything except timestamp + ds_index)
        ignore_cols = {ds_time_column, "ds_index"}
        wl_cols = [c for c in ds.columns if c not in ignore_cols]

        if len(wl_cols) == 0:
            raise ValueError("No wavelength columns found. CSV must include spectral columns besides timestamp.")

        # Keep a consistent wavelength order (attempt numeric sort, else keep as-is)
        def _try_float(x):
            try:
                return float(x)
            except Exception:
                return None

        wl_vals = [_try_float(c) for c in wl_cols]
        if all(v is not None for v in wl_vals):
            wl_cols = [c for _, c in sorted(zip(wl_vals, wl_cols))]
            self.wavelengths = np.array(sorted(wl_vals), dtype=np.float32)
        else:
            self.wavelengths = np.array(wl_cols, dtype=object)

        # Extract spectra matrix (N, L)
        spectra = ds[wl_cols].to_numpy(dtype=np.float64, copy=True)

        # 3) Normalization (robust for zeros / nonpositive values)
        # Replace nonpositive with per-wavelength minimum positive (or eps) before log10
        if apply_log10:
            spectra_pos = spectra.copy()
            spectra_pos[spectra_pos <= 0] = np.nan

            # per-wavelength minimum positive
            col_min = np.nanmin(spectra_pos, axis=0)
            col_min = np.where(np.isfinite(col_min), col_min, eps)
            fill_vals = np.maximum(col_min, eps)

            # fill nonpositive/NaN
            bad = ~np.isfinite(spectra_pos)
            spectra_pos[bad] = np.take(fill_vals, np.where(bad)[1])

            spectra = np.log10(spectra_pos)

        # Normalize either per-wavelength or globally (template does global-ish for scalar;
        # for spectra, per-wavelength is typically better behaved)
        if normalize_per_wavelength:
            col_min = np.min(spectra, axis=0)
            col_std = np.std(spectra, axis=0)
            col_std = np.where(col_std > 0, col_std, 1.0)  # avoid division by zero
            spectra_norm = (spectra - col_min[None, :]) / (2.0 * col_std[None, :])
        else:
            gmin = float(np.min(spectra))
            gstd = float(np.std(spectra)) if float(np.std(spectra)) > 0 else 1.0
            spectra_norm = (spectra - gmin) / (2.0 * gstd)

        spectra_norm = spectra_norm.astype(np.float32, copy=False)

        # Store normalized spectra as an object column (one np.ndarray per row)
        # This makes merge_asof simple and keeps __getitem__ fast.
        ds["normalized_spectra"] = [spectra_norm[i, :] for i in range(spectra_norm.shape[0])]

        self.ds_index = ds  # keep full DS index

        # 4) Create Surya valid indices and find closest match to DS index
        self.df_valid_indices = pd.DataFrame({"valid_indices": self.valid_indices}).sort_values("valid_indices")
        self.df_valid_indices = pd.merge_asof(
            self.df_valid_indices,
            self.ds_index,
            right_on="ds_index",
            left_on="valid_indices",
            direction=ds_match_direction,
        )

        # 5) Remove duplicates keeping closest match (same logic as template)
        self.df_valid_indices["index_delta"] = np.abs(
            self.df_valid_indices["valid_indices"] - self.df_valid_indices["ds_index"]
        )
        self.df_valid_indices = self.df_valid_indices.sort_values(["ds_index", "index_delta"])
        self.df_valid_indices.drop_duplicates(subset="ds_index", keep="first", inplace=True)

        # Enforce tolerance
        if ds_time_tolerance is not None:
            self.df_valid_indices = self.df_valid_indices.loc[
                self.df_valid_indices["index_delta"] <= pd.Timedelta(ds_time_tolerance),
                :,
            ]
            if len(self.df_valid_indices) == 0:
                raise ValueError("No intersection between Surya and DS indices within ds_time_tolerance")

        # Override valid indices to reflect matches
        self.valid_indices = [pd.Timestamp(d) for d in self.df_valid_indices["valid_indices"]]
        self.adjusted_length = len(self.valid_indices)
        self.df_valid_indices.set_index("valid_indices", inplace=True)

        if max_number_of_samples is not None:
            self.adjusted_length = min(self.adjusted_length, max_number_of_samples)

    def __len__(self):
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict containing (optionally) Surya stack + downstream forecast
            - forecast: np.ndarray, shape (n_wavelengths,), dtype float32
            - ds_index: ISO timestamp string of the matched downstream row
        """
        base_dictionary = {}

        if self.return_surya_stack:
            base_dictionary = super().__getitem__(idx=idx)

        # IMPORTANT: forecast is a 1D array for your case
        spec = self.df_valid_indices.iloc[idx]["normalized_spectra"]
        base_dictionary["forecast"] = np.asarray(spec, dtype=np.float32)

        # Timestamp of matched DS row
        base_dictionary["ds_index"] = self.df_valid_indices["ds_index"].iloc[idx].isoformat()

        return base_dictionary
