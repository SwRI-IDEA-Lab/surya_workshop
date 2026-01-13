import h5py
import numpy as np
import pandas as pd
from typing import Literal
from workshop_infrastructure.datasets.helio_aws import HelioNetCDFDatasetAWS

class DstDataset(HelioNetCDFDatasetAWS):
    """
    Dataset class for Dst prediction.
    Inherits from HelioNetCDFDatasetAWS to load solar images (Surya stack),
    and augments them with Dst historical and target windows from an HDF5 file.

    Parameters
    ------------------
    dst_hdf5_path : str
        Path to the HDF5 file containing 'Time', 'Dst_pred1', 'Dst_per1' datasets.
    delay_days : int
        The delay used in the HDF5 generation (default 1).
    ... (Standard HelioNetCDFDatasetAWS parameters)
    """

    def __init__(
        self,
        # --- Parent Args ---
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
        # --- Dst Specific Args ---
        dst_hdf5_path: str | None = None,
        delay_days: int = 1,
        return_surya_stack: bool = True,
        max_number_of_samples: int | None = None,
    ):
        
        # 1. Initialize Parent (Loads Solar Image Logic)
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
        
        if dst_hdf5_path is None:
            raise ValueError("dst_hdf5_path must be provided")

        # 2. Load Dst Data from HDF5
        print(f"Loading Dst data from {dst_hdf5_path}...")
        try:
            with h5py.File(dst_hdf5_path, 'r') as f:
                # Load Time (GONG Time from the HDF5)
                # Assuming Time is stored as [Year, Month, Day, Hour, Minute] or similar
                # If it's the raw integer array from previous script:
                gong_time_data = np.array(f['Time'])
                
                # Load Data Windows
                # pred = History (Input), per = Future (Target)
                pred_key = f'Dst_pred{delay_days}'
                per_key = f'Dst_per{delay_days}'
                
                if pred_key not in f or per_key not in f:
                    raise KeyError(f"Could not find {pred_key} or {per_key} in HDF5 file.")
                
                self.dst_history_data = np.array(f[pred_key])
                self.dst_target_data = np.array(f[per_key])
                
                # Load Ap if available (optional, based on your request)
                # ap_pred_key = f'ap_pred{delay_days}'
                # if ap_pred_key in f:
                #     self.ap_history_data = np.array(f[ap_pred_key])

        except Exception as e:
            raise RuntimeError(f"Failed to load Dst HDF5 file: {e}")

        # 3. Create Timestamp Index for Dst Data
        # We need to match the HDF5 rows to the parent class's self.valid_indices
        dst_times = pd.to_datetime(pd.DataFrame(gong_time_data[:, :5], 
                                                columns=['Year', 'Month', 'Day', 'Hour', 'Minute']))
        
        # Create a lookup dictionary: Timestamp -> Index in HDF5 arrays
        # This is safer than assuming rows match perfectly if any shuffling/splitting happened
        self.dst_lookup = pd.Series(index=dst_times, data=np.arange(len(dst_times)))

        # 4. Filter Valid Indices
        # Only keep indices that exist in both the Solar Image Index and the Dst HDF5
        valid_timestamps_set = set(self.valid_indices)
        dst_timestamps_set = set(dst_times)
        
        # Intersection
        common_timestamps = valid_timestamps_set.intersection(dst_timestamps_set)
        
        if len(common_timestamps) == 0:
            raise ValueError("No intersection found between Solar Image timestamps and Dst HDF5 timestamps!")
        
        # Update self.valid_indices to only include the intersection
        # Sort them to keep order deterministic
        self.valid_indices = sorted(list(common_timestamps))
        self.adjusted_length = len(self.valid_indices)

        if max_number_of_samples is not None:
            self.adjusted_length = min(self.adjusted_length, max_number_of_samples)
            self.valid_indices = self.valid_indices[:self.adjusted_length]

        print(f"DstDataset initialized. matched {self.adjusted_length} samples.")

    def __len__(self):
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        base_dictionary = {}
        
        # 1. Get Solar Images (Surya Stack)
        if self.return_surya_stack:
            base_dictionary = super().__getitem__(idx=idx)

        # 2. Lookup Dst Data Indices
        sample_timestamp = self.valid_indices[idx]
        hdf5_idx = self.dst_lookup.loc[sample_timestamp]

        # 3. Extract Target Data & FORCE SHAPE
        # We fetch the raw array which might be [3, 216] or [2, 216]
        raw_target = self.dst_target_data[hdf5_idx]
        
        # --- CRITICAL FIX START ---
        # If the data has a "variable" dimension (shape > 1D), take ONLY index 0 (Dst)
        if raw_target.ndim == 2:
            # Shape is [Variables, Time]. We take row 0.
            dst_target = raw_target[0, :].astype(np.float32)
        else:
            # Shape is already [Time]. Just cast type.
            dst_target = raw_target.astype(np.float32)

        # Apply same logic to History (Input)
        raw_history = self.dst_history_data[hdf5_idx]
        if raw_history.ndim == 2:
            dst_history = raw_history[0, :].astype(np.float32)
        else:
            dst_history = raw_history.astype(np.float32)
        # --- CRITICAL FIX END ---

        # 4. Populate Dictionary
        base_dictionary["dst_history"] = dst_history  # Shape: (216,)
        base_dictionary["dst_target"] = dst_target    # Shape: (216,)
        
        # "forecast" is what the model trains against
        base_dictionary["forecast"] = dst_target
        base_dictionary["ds_index"] = sample_timestamp.isoformat()

        return base_dictionary