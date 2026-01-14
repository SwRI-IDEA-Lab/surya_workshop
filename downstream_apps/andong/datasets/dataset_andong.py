import s3fs
import h5py
import numpy as np
import pandas as pd
from workshop_infrastructure.datasets.helio_aws import HelioNetCDFDatasetAWS
# Ensure you import HelioNetCDFDatasetAWS from your module

class DstDataset(HelioNetCDFDatasetAWS):
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
        s3_use_simplecache: bool = False,
        s3_cache_dir: str = "/tmp/helio_s3_cache",
        # --- Dst Specific Args ---
        dst_hdf5_path: str | None = None,
        delay_days: int = 1,
        return_surya_stack: bool = True,
        max_number_of_samples: int | None = None,
        # --- Event Selection ---
        storm_threshold: float | None = None,
    ):
        
        # 1. Initialize Parent with anon=True
        # We explicitly tell the parent class to use anonymous mode
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
            
            # CRITICAL: Tell the parent to use anonymous access!
            # storage_options={"anon": True} 
            # Use "s3fs_kwargs" instead:
            s3fs_kwargs={"anon": True}
        )

        self.return_surya_stack = return_surya_stack
        
        if dst_hdf5_path is None:
            raise ValueError("dst_hdf5_path must be provided")

        # 2. Load Dst Data
        print(f"Loading Dst data from {dst_hdf5_path}...")
        try:
            # Handle S3 vs Local Dst file
            if dst_hdf5_path.startswith("s3://"):
                # Use anon=True here too
                if not hasattr(self, 'fs') or self.fs is None:
                    self.fs = s3fs.S3FileSystem(anon=True, client_kwargs={'region_name': 'us-west-2'})
                
                # Open S3 file
                f_obj = self.fs.open(dst_hdf5_path, 'rb') 
                
                # Read data
                with h5py.File(f_obj, 'r') as f:
                    gong_time_data = np.array(f['Time'])
                    pred_key = f'Dst_pred{delay_days}'
                    per_key = f'Dst_per{delay_days}'
                    self.dst_history_data = np.array(f[pred_key])
                    self.dst_target_data = np.array(f[per_key])
            else:
                # Local File
                with h5py.File(dst_hdf5_path, 'r') as f:
                    gong_time_data = np.array(f['Time'])
                    pred_key = f'Dst_pred{delay_days}'
                    per_key = f'Dst_per{delay_days}'
                    self.dst_history_data = np.array(f[pred_key])
                    self.dst_target_data = np.array(f[per_key])

        except Exception as e:
            raise RuntimeError(f"Failed to load Dst HDF5 file: {e}")

        # 3. Create Timestamp Index
        dst_times = pd.to_datetime(pd.DataFrame(gong_time_data[:, :5], 
                                                columns=['Year', 'Month', 'Day', 'Hour', 'Minute']))
        self.dst_lookup = pd.Series(index=dst_times, data=np.arange(len(dst_times)))

        # 4. Filter Valid Indices
        valid_timestamps_set = set(self.valid_indices)
        dst_timestamps_set = set(dst_times)
        common_timestamps = sorted(list(valid_timestamps_set.intersection(dst_timestamps_set)))
        
        # 5. Event Selection Logic
        if storm_threshold is not None:
            print(f"Applying storm filter: Dst <= {storm_threshold} nT ...")
            storm_indices = []
            
            for ts in common_timestamps:
                hdf5_idx = self.dst_lookup.loc[ts]
                target_vector = self.dst_target_data[hdf5_idx]
                
                if target_vector.ndim == 2:
                    dst_vals = target_vector[0, :]
                else:
                    dst_vals = target_vector
                
                if np.min(dst_vals) <= storm_threshold:
                    storm_indices.append(ts)
            
            self.valid_indices = storm_indices
            print(f"  > Reduced dataset from {len(common_timestamps)} to {len(self.valid_indices)} storm events.")
        else:
            self.valid_indices = common_timestamps

        # 6. Apply Max Samples Limit
        self.adjusted_length = len(self.valid_indices)
        if max_number_of_samples is not None:
            self.adjusted_length = min(self.adjusted_length, max_number_of_samples)
            self.valid_indices = self.valid_indices[:self.adjusted_length]

    def __len__(self):
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        base_dictionary = {}
        
        # 1. Get Image Stack from Parent
        # The parent (HelioNetCDFDatasetAWS) will use anon=True (from __init__)
        # and the correct path (from the CSV index).
        if self.return_surya_stack:
            base_dictionary = super().__getitem__(idx=idx)

        # 2. Get Dst Data
        sample_timestamp = self.valid_indices[idx]
        hdf5_idx = self.dst_lookup.loc[sample_timestamp]

        # Extract Target
        raw_target = self.dst_target_data[hdf5_idx]
        if raw_target.ndim == 2:
            dst_target = raw_target[0, :].astype(np.float32)
        else:
            dst_target = raw_target.astype(np.float32)

        # Extract History
        raw_history = self.dst_history_data[hdf5_idx]
        if raw_history.ndim == 2:
            dst_history = raw_history[0, :].astype(np.float32)
        else:
            dst_history = raw_history.astype(np.float32)

        # Normalize (Dst / 100.0)
        dst_target = dst_target / 100.0 
        dst_history = dst_history / 100.0
            
        base_dictionary["dst_history"] = dst_history
        base_dictionary["dst_target"] = dst_target
        base_dictionary["forecast"] = dst_target 
        base_dictionary["ds_index"] = sample_timestamp.isoformat()

        return base_dictionary