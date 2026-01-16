"""
Dataset for Stokes profile catalog.

This dataset loads Surya-generated magnetograms and observed HMI magnetograms,
then synthesizes Stokes profiles (I, Q, U, V) for all pixels using the
Milne-Eddington forward model.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional
from pathlib import Path
import sys

# Add paths for imports
sys.path.append("../../")
sys.path.append("../../Surya")

from workshop_infrastructure.datasets.helio_aws import HelioNetCDFDatasetAWS

# Import from parent directory (template_Qin)
# Add parent directory to path to import modules from template_Qin
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from test_magnetogram_rt import compute_magnetogram_to_stokes
from magnetogram_utils import load_hmi_observed_magnetogram


class StokesProfileDataset(HelioNetCDFDatasetAWS):
    """
    Dataset for Stokes profile catalog.
    
    This dataset:
    1. Loads Surya-generated magnetogram (Bx, By, Bz) from the HelioNetCDFDataset
    2. Loads observed HMI magnetogram (Bx, By, Bz) from FITS files
    3. Synthesizes Stokes profiles (I, Q, U, V) for all pixels from both magnetograms
    4. Returns input Stokes (from Surya) and ground truth Stokes (from HMI)
    
    Parameters
    ----------
    index_path : str
        Path to Surya index CSV file
    hmi_b_dir : str
        Directory containing HMI B FITS files (e.g., './datasets/hmi.B')
    time_delta_input_minutes : list[int]
        Input delta times to define the input stack in minutes from the present
    time_delta_target_minutes : int
        Target delta time to define the output stack on rollout in minutes from the present
    n_input_timestamps : int
        Number of input timestamps
    rollout_steps : int
        Number of rollout steps
    scalers : optional
        Scalers used to perform input data normalization
    channels : list[str] | None, optional
        Input channels to use, by default None
    phase : str, optional
        Descriptor of the phase used for this database, by default "train"
    s3_use_simplecache : bool, optional
        If True (default), use fsspec's simplecache to keep a local read-through cache
    s3_cache_dir : str, optional
        Directory used by simplecache. Default: /tmp/helio_s3_cache
    wavelengths : np.ndarray, optional
        Wavelength array for Stokes synthesis. Default: HMI range ±0.5 Å around 6173.15 Å
    pixel_batch_size : int, optional
        Number of pixels to process at once for Stokes synthesis, by default 10000
    device : str, optional
        Device to run Stokes synthesis on, by default 'cuda'
    max_number_of_samples : int | None, optional
        If provided, limits the maximum number of samples in the dataset, by default None
    """
    
    def __init__(
        self,
        #### All these lines are required by the parent HelioNetCDFDataset class
        index_path: str,
        hmi_b_dir: str,
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
        #### Put your downstream (DS) specific parameters below this line
        wavelengths: Optional[np.ndarray] = None,
        pixel_batch_size: int = 10000,
        device: str = 'cuda',
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
        
        self.hmi_b_dir = Path(hmi_b_dir)
        self.pixel_batch_size = pixel_batch_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Default wavelengths: HMI range ±0.5 Å around 6173.15 Å
        if wavelengths is None:
            self.wavelengths = np.linspace(6172.65, 6173.65, 50)
        else:
            self.wavelengths = wavelengths
        
        if max_number_of_samples is not None:
            self.adjusted_length = min(self.adjusted_length, max_number_of_samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Index of sample to load. (Pytorch standard.)
        Returns:
            Dictionary with following keys:
                # Input Stokes (from Surya magnetogram)
                stokes_input (torch.Tensor):     [4, n_wavelengths, H, W] - I, Q, U, V from Surya
                # Ground truth Stokes (from HMI magnetogram)
                forecast (torch.Tensor):         [4, n_wavelengths, H, W] - I, Q, U, V from HMI
                # Metadata
                ds_index (str):                  Timestamp string
                wavelengths (torch.Tensor):      [n_wavelengths] - Wavelength array
        """
        
        # Get base dictionary from parent class
        base_dictionary = super().__getitem__(idx=idx)
        
        # Extract magnetogram components from Surya output
        ts = base_dictionary['ts']  # C, T, H, W
        
        # Convert to tensor if it's a numpy array
        if isinstance(ts, np.ndarray):
            ts = torch.from_numpy(ts).float()
        elif not isinstance(ts, torch.Tensor):
            ts = torch.tensor(ts, dtype=torch.float32)
        
        # Find HMI magnetogram channel indices
        if self.channels is None:
            hmi_bx_idx = 9
            hmi_by_idx = 10
            hmi_bz_idx = 11
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
        ts_slice = ts[:, -1, :, :]  # [C, H, W]
        
        # Inverse transform to get physical units
        if isinstance(ts_slice, torch.Tensor):
            ts_slice_np = ts_slice.numpy()
        else:
            ts_slice_np = ts_slice
        
        ts_physical = self.inverse_transform_data(ts_slice_np)  # [C, H, W] in physical units
        
        # Extract Surya magnetogram: [3, H, W]
        magnetogram_surya = np.stack([
            ts_physical[hmi_bx_idx, :, :],  # Bx
            ts_physical[hmi_by_idx, :, :],  # By
            ts_physical[hmi_bz_idx, :, :],  # Bz
        ], axis=0)
        
        # Convert to tensor and add batch dimension: [1, 3, H, W]
        magnetogram_surya_tensor = torch.from_numpy(magnetogram_surya).float().unsqueeze(0)
        
        # Load observed HMI magnetogram
        timestamp = self.valid_indices[idx]
        try:
            magnetogram_hmi, H, W = load_hmi_observed_magnetogram(self.hmi_b_dir, timestamp)
        except Exception as e:
            print(f"Warning: Failed to load HMI magnetogram for timestamp {timestamp}: {e}")
            # Use Surya magnetogram as fallback (will result in perfect prediction)
            magnetogram_hmi = magnetogram_surya
            H, W = magnetogram_surya.shape[1:]
        
        # Convert HMI magnetogram to tensor and add batch dimension: [1, 3, H, W]
        magnetogram_hmi_tensor = torch.from_numpy(magnetogram_hmi).float().unsqueeze(0)
        
        # Ensure both magnetograms have the same spatial dimensions
        # If they differ, interpolate HMI to match Surya
        if magnetogram_surya_tensor.shape[-2:] != magnetogram_hmi_tensor.shape[-2:]:
            import torch.nn.functional as F
            target_size = magnetogram_surya_tensor.shape[-2:]
            magnetogram_hmi_tensor = F.interpolate(
                magnetogram_hmi_tensor,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        
        # Downsample magnetograms to 512x512 to reduce memory usage
        # Full resolution (4096x4096) Stokes profiles would be ~13GB per sample
        # 512x512 reduces this to ~200MB per sample
        max_spatial_size = 512
        if magnetogram_surya_tensor.shape[-1] > max_spatial_size or magnetogram_surya_tensor.shape[-2] > max_spatial_size:
            import torch.nn.functional as F
            magnetogram_surya_tensor = F.interpolate(
                magnetogram_surya_tensor,
                size=(max_spatial_size, max_spatial_size),
                mode='bilinear',
                align_corners=False
            )
            magnetogram_hmi_tensor = F.interpolate(
                magnetogram_hmi_tensor,
                size=(max_spatial_size, max_spatial_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Synthesize Stokes profiles from Surya magnetogram (input)
        # Use CPU for synthesis to avoid GPU memory issues, then keep on CPU
        stokes_surya, _ = compute_magnetogram_to_stokes(
            magnetogram_surya_tensor,
            self.wavelengths,
            device='cpu',  # Use CPU for synthesis to save GPU memory
            pixel_batch_size=self.pixel_batch_size
        )
        # stokes_surya: [1, 4, n_wavelengths, H, W]
        stokes_surya = stokes_surya.squeeze(0).cpu()  # [4, n_wavelengths, H, W] - Keep on CPU
        
        # Synthesize Stokes profiles from HMI magnetogram (ground truth)
        stokes_hmi, _ = compute_magnetogram_to_stokes(
            magnetogram_hmi_tensor,
            self.wavelengths,
            device='cpu',  # Use CPU for synthesis to save GPU memory
            pixel_batch_size=self.pixel_batch_size
        )
        # stokes_hmi: [1, 4, n_wavelengths, H, W]
        stokes_hmi = stokes_hmi.squeeze(0).cpu()  # [4, n_wavelengths, H, W] - Keep on CPU
        
        # Build return dictionary
        result_dict = {
            'stokes_input': stokes_surya,  # [4, n_wavelengths, H, W] - Input from Surya
            'forecast': stokes_hmi,  # [4, n_wavelengths, H, W] - Ground truth from HMI
            'wavelengths': torch.from_numpy(self.wavelengths).float(),  # [n_wavelengths]
            'ds_index': pd.Timestamp(timestamp).isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
        }
        
        return result_dict
