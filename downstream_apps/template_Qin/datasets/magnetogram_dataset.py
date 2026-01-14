import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Literal
from pathlib import Path
import sys

# Add paths for imports
sys.path.append("../../")
sys.path.append("../../Surya")

from workshop_infrastructure.datasets.helio_aws import HelioNetCDFDatasetAWS


class MagnetogramTestDataset(HelioNetCDFDatasetAWS):
    """
    Dataset for testing Surya-generated magnetograms against radiative transfer equation.
    
    This dataset loads Surya-generated magnetograms (hmi_bx, hmi_by, hmi_bz) and optionally
    observed Stokes profiles to validate against the Milne-Eddington radiative transfer model.
    
    Parameters
    ----------
    index_path : str
        Path to Surya index
    time_delta_input_minutes : list[int]
        Input delta times to define the input stack in minutes from the present
    time_delta_target_minutes : int
        Target delta time to define the output stack on rollout in minutes from the present
    n_input_timestamps : int
        Number of input timestamps
    rollout_steps : int
        Number of rollout steps
    scalers : optional
        scalers used to perform input data normalization, by default None
    channels : list[str] | None, optional
        Input channels to use, by default None
    phase : str, optional
        Descriptor of the phase used for this database, by default "test"
    s3_use_simplecache : bool, optional
        If True (default), use fsspec's simplecache to keep a local read-through
        cache of objects.
    s3_cache_dir : str, optional
        Directory used by simplecache. Default: /tmp/helio_s3_cache
    return_stokes : bool, optional
        If True, return observed Stokes profiles for validation, by default False
    stokes_data_path : str, optional
        Path to directory containing Stokes profile data files, by default None
    max_number_of_samples : int | None, optional
        If provided, limits the maximum number of samples in the dataset, by default None
    """
    
    def __init__(
        self,
        #### All these lines are required by the parent HelioNetCDFDataset class
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
        phase="test",
        s3_use_simplecache: bool = True,
        s3_cache_dir: str = "/tmp/helio_s3_cache",
        #### Put your downstream (DS) specific parameters below this line
        return_stokes: bool = False,
        stokes_data_path: str | None = None,
        max_number_of_samples: int | None = None,
    ):
        ## Initialize parent class
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
        
        self.return_stokes = return_stokes
        self.stokes_data_path = stokes_data_path
        
        if max_number_of_samples is not None:
            self.adjusted_length = min(self.adjusted_length, max_number_of_samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Index of sample to load. (Pytorch standard.)
        Returns:
            Dictionary with following keys. The values are tensors with shape as follows:
                # Surya keys--------------------------------
                ts (torch.Tensor):                C, T, H, W
                time_delta_input (torch.Tensor):  T
                input_latitude (torch.Tensor):    T
                lead_time_delta (torch.Tensor):   L
                forecast_latitude (torch.Tensor): L
                # Magnetogram keys-------------------------
                magnetogram (torch.Tensor):       B, 3, H, W  (Bx, By, Bz components)
                # Optional Stokes keys--------------------
                stokes_profiles (torch.Tensor):   B, 4, N_wavelengths (if return_stokes=True)
                wavelengths (torch.Tensor):      N_wavelengths (if return_stokes=True)
            C - Channels, T - Input times, H - Image height, W - Image width, 
            L - Lead time, B - Batch size, N_wavelengths - Number of wavelength points.
        """
        
        # Get base dictionary from parent class
        base_dictionary = super().__getitem__(idx=idx)
        
        # Extract magnetogram components from Surya output
        # Assuming channels are in order: ['aia94', 'aia131', ..., 'hmi_m', 'hmi_bx', 'hmi_by', 'hmi_bz', 'hmi_v']
        ts = base_dictionary['ts']  # C, T, H, W
        
        # Convert to tensor if it's a numpy array
        if isinstance(ts, np.ndarray):
            ts = torch.from_numpy(ts).float()
        elif not isinstance(ts, torch.Tensor):
            ts = torch.tensor(ts, dtype=torch.float32)
        
        # Find HMI magnetogram channel indices
        # This assumes the channels list matches the config
        if self.channels is None:
            # Default channel order
            hmi_bx_idx = 9  # hmi_bx
            hmi_by_idx = 10  # hmi_by
            hmi_bz_idx = 11  # hmi_bz
        else:
            try:
                hmi_bx_idx = self.channels.index('hmi_bx')
                hmi_by_idx = self.channels.index('hmi_by')
                hmi_bz_idx = self.channels.index('hmi_bz')
            except ValueError:
                # Fallback: assume last 3 channels before hmi_v are magnetogram components
                hmi_bx_idx = len(self.channels) - 4
                hmi_by_idx = len(self.channels) - 3
                hmi_bz_idx = len(self.channels) - 2
        
        # Extract magnetogram components (use the last time step)
        # Note: ts is normalized, we need to inverse transform to get physical units
        # Get the full ts slice for the last time step: [C, H, W]
        ts_slice = ts[:, -1, :, :]  # [C, H, W]
        
        # Inverse transform the full ts slice to get physical units
        if isinstance(ts_slice, torch.Tensor):
            ts_slice_np = ts_slice.numpy()
        else:
            ts_slice_np = ts_slice
        
        # Use parent class inverse_transform_data method
        ts_physical = self.inverse_transform_data(ts_slice_np)  # [C, H, W] in physical units
        
        # Find HMI velocity channel index
        if self.channels is None:
            hmi_v_idx = 12  # hmi_v (typically after hmi_bz)
        else:
            try:
                hmi_v_idx = self.channels.index('hmi_v')
            except ValueError:
                # Fallback: assume last channel is hmi_v
                hmi_v_idx = len(self.channels) - 1
        
        # Extract magnetogram components from inverse-transformed data
        # Shape: 3, H, W
        magnetogram = torch.stack([
            torch.from_numpy(ts_physical[hmi_bx_idx, :, :]).float(),  # Bx
            torch.from_numpy(ts_physical[hmi_by_idx, :, :]).float(),  # By
            torch.from_numpy(ts_physical[hmi_bz_idx, :, :]).float(),  # Bz
        ], dim=0)
        
        # Extract velocity (hmi_v) for Doppler shift calculation
        # Shape: H, W (in m/s)
        velocity = torch.from_numpy(ts_physical[hmi_v_idx, :, :]).float()
        
        base_dictionary['magnetogram'] = magnetogram
        base_dictionary['velocity'] = velocity  # Velocity in m/s
        
        # Optionally load Stokes profiles if available
        if self.return_stokes and self.stokes_data_path is not None:
            # Load Stokes profiles for this timestamp
            timestamp = self.valid_indices[idx]
            stokes_data = self._load_stokes_profiles(timestamp)
            if stokes_data is not None:
                base_dictionary['stokes_profiles'] = stokes_data['stokes']
                base_dictionary['wavelengths'] = stokes_data['wavelengths']
        
        return base_dictionary
    
    def _load_stokes_profiles(self, timestamp):
        """
        Load Stokes profiles for a given timestamp.
        
        Args:
            timestamp: Timestamp to load Stokes profiles for
            
        Returns:
            Dictionary with 'stokes' and 'wavelengths' keys, or None if not found
        """
        if self.stokes_data_path is None:
            return None
        
        stokes_path = Path(self.stokes_data_path)
        # Try to find matching Stokes data file
        # This is a placeholder - adjust based on your actual data structure
        # For example, if files are named like: stokes_20140101_000000.npy
        timestamp_str = pd.Timestamp(timestamp).strftime('%Y%m%d_%H%M%S')
        stokes_file = stokes_path / f"stokes_{timestamp_str}.npz"
        
        if stokes_file.exists():
            data = np.load(stokes_file)
            return {
                'stokes': torch.tensor(data['stokes'], dtype=torch.float32),
                'wavelengths': torch.tensor(data['wavelengths'], dtype=torch.float32)
            }
        
        return None
