"""
wsa_dataset.py

Dataset class for loading AIA 193 images paired with WSA map targets for re-training.

This module extends HelioNetCDFDatasetAWS to:
  - Load single AIA 193 image at CR middle day
  - Pair with WSA map targets loaded from local CSV files
  - Return image-to-image pairs for training

Key characteristics:
  - Input: Single AIA 193 image at CR middle day [1, H, W]
  - Target: WSA map [1, H, W]
  - CR numbers are provided as a list; dataset maps CR → timestamp → AIA file → WSA map
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, Optional, List, Literal
from sunpy.coordinates.sun import carrington_rotation_time
from workshop_infrastructure.datasets.helio_aws import HelioNetCDFDatasetAWS


def get_cr_middle_timestamp(cr: int) -> datetime:
    """
    Get the middle timestamp of a Carrington Rotation using sunpy.
    
    Parameters
    ----------
    cr : int
        Carrington Rotation number (e.g., 2049)
    
    Returns
    -------
    datetime
        Middle timestamp of the CR
    
    Notes
    -----
    - Uses sunpy.coordinates.sun.carrington_rotation_time to get exact CR boundaries
    - Middle time = (CR_start + CR_end) / 2
    """
    cr_start = carrington_rotation_time(cr)
    cr_end = carrington_rotation_time(cr + 1)
    
    # Convert to datetime if needed
    if hasattr(cr_start, 'datetime'):
        cr_start = cr_start.datetime
    if hasattr(cr_end, 'datetime'):
        cr_end = cr_end.datetime
    
    # Calculate middle
    middle_time = cr_start + (cr_end - cr_start) / 2
    return middle_time


def find_nearest_timestamp(cr: int, surya_timestamps: np.ndarray, tolerance_days: int = 3) -> Optional[datetime]:
    """
    Find the nearest available timestamp in Surya index for a given CR.
    
    Parameters
    ----------
    cr : int
        Carrington Rotation number
    surya_timestamps : np.ndarray
        Array of available timestamps from Surya index (as datetime objects or strings)
    tolerance_days : int, default=3
        Maximum tolerance in days to search for nearest timestamp
    
    Returns
    -------
    datetime or None
        Nearest timestamp in Surya index, or None if no match found within tolerance
    """
    # Get middle timestamp of CR using sunpy
    target_time = get_cr_middle_timestamp(cr)
    
    # Convert surya_timestamps to datetime if they're strings
    surya_timestamps = pd.to_datetime(surya_timestamps)
    
    # Find nearest timestamp within tolerance
    tolerance = pd.Timedelta(days=tolerance_days)
    
    diffs = np.abs(surya_timestamps - target_time)
    min_diff_idx = np.argmin(diffs)
    
    if diffs[min_diff_idx] <= tolerance:
        return surya_timestamps[min_diff_idx]
    
    return None


class WSAImageDataset(HelioNetCDFDatasetAWS):
    """
    Dataset for loading AIA 193 images paired with WSA map targets.
    
    Extends HelioNetCDFDatasetAWS to:
      - Take a list of Carrington Rotation (CR) numbers
      - For each CR, calculate the middle day using sunpy
      - Load the corresponding AIA 193 image from Surya S3 index
      - Load the corresponding WSA map from local CSV files
      - Return single-timestamp image-to-image pairs
    
    Parameters
    ----------
    cr_list : list[int]
        List of CR numbers to include (e.g., [2049, 2053, 2054, ...])
    
    surya_index_path : str
        Path to Surya index CSV (with columns: path, timestep, present)
    
    wsa_map_dir : str
        Directory containing WSA map CSV files (e.g., "downstream_apps/template/datasets/wsa_full_disk/")
        Files should be named: reprojected_wsa_CR{CR}.csv
    
    wsa_params_path : str
        Path to fitted_wsa_params.csv (contains vmin, vmax for normalization)
    
    scalers : dict, optional
        Pre-loaded scalers for AIA normalization (with keys: means, stds, sl_scale_factors, epsilons)
    
    channels : list[str], optional
        Channels to load (default: ["0193"] for AIA 193 only)
    
    normalize_wsa : bool, default=True
        Whether to normalize WSA maps to [0, 1] using vmin/vmax
    
    tolerance_days : int, default=3
        Maximum tolerance in days when searching for AIA timestamp for a given CR
    
    s3_use_simplecache : bool, optional
        If True (default), use fsspec's simplecache to keep a local read-through cache
    
    s3_cache_dir : str, optional
        Directory used by simplecache. Default: /tmp/helio_s3_cache
    
    phase : str, optional
        Descriptor of the phase used for this database, by default "train"
    """
    
    def __init__(
        self,
        cr_list: List[int],
        surya_index_path: str,
        wsa_map_dir: str,
        wsa_params_path: str,
        scalers: Optional[Dict] = None,
        channels: Optional[List[str]] = None,
        normalize_wsa: bool = True,
        tolerance_days: int = 3,
        s3_use_simplecache: bool = True,
        s3_cache_dir: str = "/tmp/helio_s3_cache",
        phase: str = "train",
    ):
        self.cr_list = cr_list
        self.wsa_map_dir = wsa_map_dir
        self.normalize_wsa = normalize_wsa
        self.tolerance_days = tolerance_days
        
        # Load Surya index
        self.surya_index = pd.read_csv(surya_index_path)
        self.surya_index["timestep"] = pd.to_datetime(self.surya_index["timestep"])
        self.surya_index = self.surya_index[self.surya_index["present"] == 1]
        self.surya_index.sort_values("timestep", inplace=True)
        
        # Load WSA parameters for normalization
        self.wsa_params = pd.read_csv(wsa_params_path, index_col="CR")
        
        # Initialize parent class with n_input_timestamps=1 (single frame, no temporal stack)
        # We use dummy values for temporal parameters since we only need one timestamp
        super().__init__(
            index_path=surya_index_path,
            time_delta_input_minutes=[0],  # Single timestamp only
            time_delta_target_minutes=0,   # No forecast target
            n_input_timestamps=1,          # Single frame
            rollout_steps=1,               # No rollout
            scalers=scalers,
            num_mask_aia_channels=0,
            drop_hmi_probability=0,
            use_latitude_in_learned_flow=False,
            channels=channels if channels is not None else ["0193"],
            phase=phase,
            s3_use_simplecache=s3_use_simplecache,
            s3_cache_dir=s3_cache_dir,
        )

        # Build valid samples (CR → nearest timestamp mapping)
        self.samples = []  # List of (cr, timestamp, path_to_aia, path_to_wsa_map)
        self._build_samples()
        
    def _build_samples(self):
        """Build list of valid CR samples with corresponding timestamps and paths."""
        for cr in self.cr_list:
            # Find nearest timestamp in Surya index for this CR
            nearest_ts = find_nearest_timestamp(
                cr, 
                self.surya_index["timestep"].values,
                tolerance_days=self.tolerance_days
            )
            
            if nearest_ts is None:
                print(f"Warning: No AIA data found for CR {cr} within {self.tolerance_days} days")
                continue
            
            # Get path to AIA file
            aia_row = self.surya_index[self.surya_index["timestep"] == nearest_ts].iloc[0]
            aia_path = aia_row["path"]
            
            # Check if WSA map CSV exists
            wsa_map_path = os.path.join(self.wsa_map_dir, f"reprojected_wsa_CR{cr}.csv")
            if not os.path.exists(wsa_map_path):
                print(f"Warning: WSA map not found for CR {cr} at {wsa_map_path}")
                continue
            
            # Check if CR params exist
            if cr not in self.wsa_params.index:
                print(f"Warning: WSA params not found for CR {cr}")
                continue
            
            self.samples.append((cr, nearest_ts, aia_path, wsa_map_path))
        
        print(f"Loaded {len(self.samples)} valid CR samples")
        
        # Override parent's valid_indices to use only our CR-matched samples
        self.valid_indices = [pd.Timestamp(ts) for cr, ts, _, _ in self.samples]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Load and return an image-to-image pair.
        
        Parameters
        ----------
        idx : int
            Index in the dataset
        
        Returns
        -------
        dict
            Dictionary with keys:
              - "ts": AIA 193 image [C=1, T=1, H, W] (from parent class)
              - "wsa_map": WSA map [1, H, W] normalized to [0, 1]
              - "carrington_rotation": CR number
              - "timestamp": timestamp of the AIA image (ISO format)
        """
        cr, timestamp, aia_path, wsa_map_path = self.samples[idx]
        
        # Get AIA data from parent class
        # This will return a dict with "ts" key containing [C, T, H, W]
        base_dict = super().__getitem__(idx)
        
        # Load WSA map from CSV
        wsa_map = self._load_wsa_map_from_csv(wsa_map_path)  # [H, W]
        
        # Normalize WSA map if requested
        if self.normalize_wsa:
            params = self.wsa_params.loc[cr]
            vmin, vmax = params["vmin"], params["vmax"]
            wsa_map = (wsa_map - vmin) / (vmax - vmin + 1e-8)
            wsa_map = np.clip(wsa_map, 0, 1)
        
        # Add channel dimension if needed
        if wsa_map.ndim == 2:
            wsa_map = wsa_map[np.newaxis, :, :]  # [1, H, W]
        
        # Add WSA-specific fields to the returned dictionary
        base_dict["wsa_map"] = torch.from_numpy(wsa_map).float()
        base_dict["carrington_rotation"] = cr
        base_dict["timestamp"] = timestamp.isoformat()
        
        return base_dict
    
    def _load_wsa_map_from_csv(self, csv_path: str) -> np.ndarray:
        """
        Load WSA map from CSV file and reshape to [H, W] = [4096, 4096].
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file containing flattened WSA map
        
        Returns
        -------
        np.ndarray
            WSA map with shape [4096, 4096]
        """
        try:
            # Load CSV - assuming it contains a single column of pixel values or 2D array
            df = pd.read_csv(csv_path)
            
            # Convert to numpy array
            if df.shape[1] == 1:
                # Single column: flatten 1D array
                wsa_flat = df.iloc[:, 0].values
                wsa_map = wsa_flat.reshape(4096, 4096)
            else:
                # Multi-column: assume it's already in 2D format or needs transpose
                wsa_map = df.values
                if wsa_map.shape != (4096, 4096):
                    # Try reshaping if dimensions don't match
                    total_pixels = wsa_map.size
                    if total_pixels == 4096 * 4096:
                        wsa_map = wsa_map.reshape(4096, 4096)
                    else:
                        raise ValueError(
                            f"WSA map has {total_pixels} pixels, "
                            f"expected {4096*4096}. Cannot reshape to [4096, 4096]"
                        )
            
            return wsa_map.astype(np.float32)
        
        except Exception as e:
            raise RuntimeError(f"Failed to load WSA map from {csv_path}: {e}")