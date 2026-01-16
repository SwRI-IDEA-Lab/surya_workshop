"""
Utility functions for magnetogram testing and Stokes profile synthesis.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import sys
import glob
from astropy.io import fits
import pandas as pd

sys.path.append("../../")
sys.path.append("../../Surya")

from ME_PINN_legacy.me_pinn_hmi import MEPhysicsLoss
from datasets.magnetogram_dataset import MagnetogramTestDataset
from surya.utils.data import build_scalers


def load_magnetogram_sample(config_path, sample_idx=0):
    """
    Load a single magnetogram sample from Surya dataset.
    
    Args:
        config_path: Path to configuration YAML file
        sample_idx: Index of sample to load (default: 0)
        
    Returns:
        magnetogram_np: Numpy array of shape [3, H, W] with Bx, By, Bz in physical units (Gauss)
        velocity_np: Numpy array of shape [H, W] with velocity in m/s (or None if not available)
        timestamp: Timestamp (pandas Timestamp) for this sample
        dataset: The dataset object
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config["data"]["scalers"] = yaml.safe_load(open(config["data"]["scalers_path"], "r"))
    scalers = build_scalers(info=config["data"]["scalers"])
    
    # Create dataset (load just one sample)
    dataset = MagnetogramTestDataset(
        index_path=config["data"]["valid_data_path"],
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        n_input_timestamps=config["data"].get("n_input_timestamps", 1),
        rollout_steps=1,
        scalers=scalers,
        channels=config["data"]["channels"],
        phase="test",
        return_stokes=False,
        max_number_of_samples=sample_idx + 1,
    )
    
    # Get timestamp from dataset valid_indices
    timestamp = dataset.valid_indices[sample_idx]
    
    # Load sample (magnetogram is already inverse-transformed in the dataset)
    sample = dataset[sample_idx]
    magnetogram = sample['magnetogram']  # [3, H, W] in physical units (Gauss)
    
    # Convert to numpy if tensor
    if torch.is_tensor(magnetogram):
        magnetogram_np = magnetogram.numpy()
    else:
        magnetogram_np = magnetogram
    
    # Extract velocity if available
    velocity_np = None
    if 'velocity' in sample:
        velocity = sample['velocity']  # [H, W] in m/s
        if torch.is_tensor(velocity):
            velocity_np = velocity.numpy()
        else:
            velocity_np = velocity
    
    return magnetogram_np, velocity_np, timestamp, dataset


def velocity_to_lambda0(velocity, lambda_rest=6173.15):
    """
    Convert velocity to wavelength shift using Doppler effect.
    
    Args:
        velocity: Velocity in m/s (positive = redshift, negative = blueshift)
        lambda_rest: Rest wavelength in Angstroms (default: 6173.15 for HMI)
        
    Returns:
        lambda0: Wavelength shift in Angstroms
    """
    c = 299792458.0  # Speed of light in m/s
    lambda0 = lambda_rest * velocity / c
    return lambda0


def extract_pixel_bxyz(magnetogram_np, velocity_np=None, pixel_h=None, pixel_w=None, use_max_field=False):
    """
    Extract Bx, By, Bz, and velocity for a single pixel from magnetogram.
    
    Args:
        magnetogram_np: Numpy array of shape [3, H, W] with Bx, By, Bz
        velocity_np: Optional numpy array of shape [H, W] with velocity in m/s
        pixel_h: Row index of pixel (if None, uses center)
        pixel_w: Column index of pixel (if None, uses center)
        use_max_field: If True, use pixel with maximum field strength
        
    Returns:
        Bx, By, Bz, B_mag: Field components and magnitude in Gauss
        velocity: Velocity in m/s (or None if not provided)
        pixel_h, pixel_w: Pixel coordinates
    """
    H, W = magnetogram_np.shape[1], magnetogram_np.shape[2]
    
    if use_max_field:
        B_mag_2d = np.sqrt(magnetogram_np[0]**2 + magnetogram_np[1]**2 + magnetogram_np[2]**2)
        max_B_idx = np.unravel_index(np.argmax(B_mag_2d), B_mag_2d.shape)
        pixel_h, pixel_w = max_B_idx[0], max_B_idx[1]
    else:
        if pixel_h is None:
            pixel_h = H // 2
        if pixel_w is None:
            pixel_w = W // 2
    
    Bx = float(magnetogram_np[0, pixel_h, pixel_w])
    By = float(magnetogram_np[1, pixel_h, pixel_w])
    Bz = float(magnetogram_np[2, pixel_h, pixel_w])
    B_mag = float(np.sqrt(Bx**2 + By**2 + Bz**2))
    
    velocity = None
    if velocity_np is not None:
        velocity = float(velocity_np[pixel_h, pixel_w])
    
    return Bx, By, Bz, B_mag, velocity, pixel_h, pixel_w


def bxyz_to_spherical(Bx, By, Bz):
    """
    Convert Bx, By, Bz to spherical coordinates (B, theta, chi).
    
    Args:
        Bx, By, Bz: Magnetic field components in Gauss
        
    Returns:
        B_mag: Field magnitude
        theta: Inclination angle [0, π]
        chi: Azimuth angle [0, π]
    """
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2 + 1e-6)  # Add epsilon to avoid division by zero
    theta = np.arccos(np.clip(Bz / B_mag, -1.0, 1.0))  # Inclination [0, π]
    chi = np.arctan2(By, Bx)  # Azimuth
    chi = (chi + np.pi) % (2 * np.pi)  # Normalize to [0, 2π]
    chi = chi if chi <= np.pi else chi - np.pi  # Normalize to [0, π]
    
    return B_mag, theta, chi


def synthesize_stokes_single_pixel(
    Bx, By, Bz,
    lambda_rest=6173.15,
    geff=2.5,
    wavelengths=None,
    eta0=2.0,
    dlambdaD=0.15,
    a=0.1,
    lambda0=None,
    velocity=None,
    B0=1.0,
    B1=0.5,
    device='cuda'
):
    """
    Synthesize Stokes profiles (I, Q, U, V) for a single pixel from Bx, By, Bz.
    
    Args:
        Bx, By, Bz: Magnetic field components in Gauss
        lambda_rest: Rest wavelength in Angstroms (default: 6173.15 for HMI)
        geff: Effective Landé factor (default: 2.5 for HMI)
        wavelengths: Wavelength array (default: ±0.5 Å around lambda_rest)
        eta0: Line-to-continuum opacity ratio (default: 2.0)
        dlambdaD: Doppler width in Angstroms (default: 0.15)
        a: Damping parameter (default: 0.1)
        lambda0: Line center shift in Angstroms (default: 0.0, or computed from velocity if provided)
        velocity: Velocity in m/s (if provided, lambda0 will be computed from this)
        B0: Continuum source function (default: 1.0)
        B1: Line source function (default: 0.5)
        device: Device to run computation on (default: 'cuda')
        
    Returns:
        stokes_pred: Numpy array of shape [4, n_wavelengths] with I, Q, U, V
        wavelengths: Wavelength array used
        me_params: Dictionary with ME parameters used
    """
    # Compute lambda0 from velocity if provided, otherwise use provided lambda0 or default to 0.0
    if velocity is not None:
        lambda0 = velocity_to_lambda0(velocity, lambda_rest)
    elif lambda0 is None:
        lambda0 = 0.0
    # Default wavelength array
    if wavelengths is None:
        wavelengths = np.linspace(lambda_rest - 0.5, lambda_rest + 0.5, 50)
    
    # Convert to spherical coordinates
    B_mag, theta, chi = bxyz_to_spherical(Bx, By, Bz)
    
    # Initialize ME physics loss function
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    physics_loss_fn = MEPhysicsLoss(lambda_rest=lambda_rest, geff=geff).to(device)
    
    # Construct ME parameters tensor
    params = torch.tensor([[B_mag, theta, chi, eta0, dlambdaD, a, lambda0, B0, B1]], 
                          dtype=torch.float32).to(device)
    
    # Ensure wavelengths is a numpy array, then convert to tensor
    if isinstance(wavelengths, torch.Tensor):
        wavelengths_tensor = wavelengths.to(device)
    else:
        wavelengths_tensor = torch.tensor(wavelengths, dtype=torch.float32).to(device)
    
    # Ensure wavelengths is 1D
    if wavelengths_tensor.dim() > 1:
        wavelengths_tensor = wavelengths_tensor.flatten()
    
    # Synthesize Stokes profiles
    with torch.no_grad():
        _, stokes_pred = physics_loss_fn(params, wavelengths_tensor, None)
    
    # Move back to CPU for plotting
    stokes_pred = stokes_pred.cpu().numpy()[0]  # [4, n_wavelengths]
    
    me_params = {
        'B': B_mag,
        'theta': theta,
        'chi': chi,
        'eta0': eta0,
        'dlambdaD': dlambdaD,
        'a': a,
        'lambda0': lambda0,
        'B0': B0,
        'B1': B1
    }
    
    # Add velocity information if it was used
    if velocity is not None:
        me_params['velocity'] = velocity
        me_params['lambda0_source'] = 'velocity'
    else:
        me_params['lambda0_source'] = 'constant'
    
    return stokes_pred, wavelengths, me_params


def plot_stokes_profiles(stokes_pred, wavelengths, Bx, By, Bz, B_mag, pixel_h=None, pixel_w=None, 
                         lambda_rest=6173.15, save_path=None):
    """
    Plot synthesized Stokes profiles in separate subplots.
    
    Args:
        stokes_pred: Numpy array of shape [4, n_wavelengths] with I, Q, U, V
        wavelengths: Wavelength array
        Bx, By, Bz, B_mag: Field components and magnitude
        pixel_h, pixel_w: Pixel coordinates (optional, for title)
        lambda_rest: Rest wavelength for vertical line
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = ['black', 'red', 'green', 'blue']
    stokes_names = ['I', 'Q', 'U', 'V']
    
    # Create title string
    title_base = f'Synthesized Stokes Profiles'
    if pixel_h is not None and pixel_w is not None:
        title_base += f' (Pixel at ({pixel_h}, {pixel_w}))'
    title_base += f'\nBx={Bx:.1f} G, By={By:.1f} G, Bz={Bz:.1f} G, |B|={B_mag:.1f} G'
    
    fig.suptitle(title_base, fontsize=14, y=0.995)
    
    for i, (name, color, ax) in enumerate(zip(stokes_names, colors, axes)):
        ax.plot(wavelengths, stokes_pred[i], color=color, linewidth=2, alpha=0.8)
        ax.set_xlabel('Wavelength (Å)', fontsize=11)
        ax.set_ylabel(f'Stokes {name}', fontsize=11)
        ax.set_title(f'Stokes {name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axvline(lambda_rest, color='gray', linestyle=':', alpha=0.5, 
                  label=f'Line center ({lambda_rest} Å)')
        ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for suptitle
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def load_hmi_b_parameters(hmi_b_dir, timestamp, pixel_h=None, pixel_w=None):
    """
    Load HMI B inversion parameters (field, inclination, azimuth, eta_0, dop_width, damping) 
    from FITS files for a given timestamp and pixel.
    
    Args:
        hmi_b_dir: Directory containing HMI B FITS files (e.g., './datasets/hmi.B')
        timestamp: The timestamp (pandas Timestamp or numpy datetime64) to match.
                  Can also be a string in format 'YYYYMMDD_HHMMSS' or 'HHMMSS' (assumes same date).
        pixel_h, pixel_w: Pixel coordinates (optional). If None, uses center (2048, 2048).
    
    Returns:
        params_dict: Dictionary with parameters:
            - B_mag: Magnetic field strength (Gauss)
            - theta: Inclination angle (radians)
            - chi: Azimuth angle (radians)
            - Bx, By, Bz: Field components (Gauss)
            - eta0: Line-to-continuum opacity ratio
            - dlambdaD: Doppler width (Angstroms)
            - a: Damping parameter
        pixel_h, pixel_w: Actual pixel coordinates used
    """
    # Convert timestamp to string format YYYYMMDD_HHMMSS
    if isinstance(timestamp, pd.Timestamp):
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
    elif isinstance(timestamp, np.datetime64):
        timestamp_str = pd.Timestamp(timestamp).strftime('%Y%m%d_%H%M%S')
    elif isinstance(timestamp, str):
        # If only HHMMSS provided, assume same date as first file
        if len(timestamp) == 6 and '_' not in timestamp:
            # Find first file to get date
            files = glob.glob(f"{hmi_b_dir}/hmi.b_720s.*.field.fits")
            if files:
                first_file = Path(files[0]).name
                date_str = first_file.split('_')[2]  # e.g., '20110116'
                timestamp_str = f"{date_str}_{timestamp}"
            else:
                raise ValueError(f"No HMI B files found in {hmi_b_dir}")
        else:
            timestamp_str = timestamp
    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")
    
    # Default pixel location
    if pixel_h is None or pixel_w is None:
        pixel_h, pixel_w = 2048, 2048
    
    # Construct file paths
    base_filename = f"hmi.b_720s.{timestamp_str}_TAI"
    field_file = Path(hmi_b_dir) / f"{base_filename}.field.fits"
    inclination_file = Path(hmi_b_dir) / f"{base_filename}.inclination.fits"
    azimuth_file = Path(hmi_b_dir) / f"{base_filename}.azimuth.fits"
    eta_0_file = Path(hmi_b_dir) / f"{base_filename}.eta_0.fits"
    dop_width_file = Path(hmi_b_dir) / f"{base_filename}.dop_width.fits"
    damping_file = Path(hmi_b_dir) / f"{base_filename}.damping.fits"
    
    # Check if files exist
    files_to_check = [field_file, inclination_file, azimuth_file, eta_0_file, dop_width_file, damping_file]
    missing_files = [f for f in files_to_check if not f.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing HMI B files for timestamp {timestamp_str}: {[f.name for f in missing_files]}")
    
    # Load pixel values
    def load_pixel_value(fits_file):
        with fits.open(fits_file) as hdul:
            data = hdul[1].data if hdul[1].data is not None else hdul[0].data
            return float(data[pixel_h, pixel_w])
    
    # Load parameters
    B_mag = load_pixel_value(field_file)  # Field strength in Gauss
    theta_deg = load_pixel_value(inclination_file)  # Inclination in degrees
    chi_deg = load_pixel_value(azimuth_file)  # Azimuth in degrees
    eta0 = load_pixel_value(eta_0_file)  # Eta_0 parameter
    dlambdaD = load_pixel_value(dop_width_file)  # Doppler width in Angstroms
    a = load_pixel_value(damping_file)  # Damping parameter
    
    # Convert angles to radians
    theta = np.deg2rad(theta_deg)
    chi = np.deg2rad(chi_deg)
    
    # Convert spherical coordinates to Cartesian
    Bx = B_mag * np.sin(theta) * np.cos(chi)
    By = B_mag * np.sin(theta) * np.sin(chi)
    Bz = B_mag * np.cos(theta)
    
    params_dict = {
        'B_mag': B_mag,
        'theta': theta,
        'chi': chi,
        'Bx': Bx,
        'By': By,
        'Bz': Bz,
        'eta0': eta0,
        'dlambdaD': dlambdaD,
        'a': a
    }
    
    return params_dict, pixel_h, pixel_w


def load_hmi_observed_magnetogram(hmi_b_dir, timestamp):
    """
    Load full observed magnetogram (Bx, By, Bz) from HMI B FITS files for a given timestamp.
    
    Args:
        hmi_b_dir: Directory containing HMI B FITS files (e.g., './datasets/hmi.B')
        timestamp: The timestamp (pandas Timestamp or numpy datetime64) to match.
                  Can also be a string in format 'YYYYMMDD_HHMMSS' or 'HHMMSS' (assumes same date).
    
    Returns:
        magnetogram_obs: Numpy array of shape [3, H, W] with Bx, By, Bz in Gauss
        H, W: Height and width of the magnetogram
    """
    # Convert timestamp to string format YYYYMMDD_HHMMSS
    if isinstance(timestamp, pd.Timestamp):
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
    elif isinstance(timestamp, np.datetime64):
        timestamp_str = pd.Timestamp(timestamp).strftime('%Y%m%d_%H%M%S')
    elif isinstance(timestamp, str):
        # If only HHMMSS provided, assume same date as first file
        if len(timestamp) == 6 and '_' not in timestamp:
            # Find first file to get date
            files = glob.glob(f"{hmi_b_dir}/hmi.b_720s.*.field.fits")
            if files:
                first_file = Path(files[0]).name
                date_str = first_file.split('_')[2]  # e.g., '20110116'
                timestamp_str = f"{date_str}_{timestamp}"
            else:
                raise ValueError(f"No HMI B files found in {hmi_b_dir}")
        else:
            timestamp_str = timestamp
    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")
    
    # Construct file paths
    base_filename = f"hmi.b_720s.{timestamp_str}_TAI"
    field_file = Path(hmi_b_dir) / f"{base_filename}.field.fits"
    inclination_file = Path(hmi_b_dir) / f"{base_filename}.inclination.fits"
    azimuth_file = Path(hmi_b_dir) / f"{base_filename}.azimuth.fits"
    disambig_file = Path(hmi_b_dir) / f"{base_filename}.disambig.fits"
    
    # Check if files exist
    files_to_check = [field_file, inclination_file, azimuth_file]
    missing_files = [f for f in files_to_check if not f.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing HMI B files for timestamp {timestamp_str}: {[f.name for f in missing_files]}")
    
    # Load full data arrays
    with fits.open(field_file) as hdul:
        B_mag_data = hdul[1].data if hdul[1].data is not None else hdul[0].data
        H, W = B_mag_data.shape
    
    with fits.open(inclination_file) as hdul:
        inclination_data = hdul[1].data if hdul[1].data is not None else hdul[0].data
    
    with fits.open(azimuth_file) as hdul:
        azimuth_data = hdul[1].data if hdul[1].data is not None else hdul[0].data
    
    # Load disambiguation file if available
    disambig_data = None
    if disambig_file.exists():
        with fits.open(disambig_file) as hdul:
            disambig_data = hdul[1].data if hdul[1].data is not None else hdul[0].data
        print(f"Loaded disambiguation file for timestamp {timestamp_str}")
    else:
        print(f"Warning: Disambiguation file not found for timestamp {timestamp_str}, proceeding without disambiguation")
    
    # Convert angles to radians
    theta = np.deg2rad(inclination_data)  # Inclination in degrees -> radians
    chi = np.deg2rad(azimuth_data)  # Azimuth in degrees -> radians
    
    # Apply disambiguation: if bit 0 is set, add 180 degrees to azimuth
    if disambig_data is not None:
        # Extract bit 0 (lowest bit) - this indicates whether to add 180 degrees
        # Convert to uint8 if needed to ensure proper bit operations
        disambig_uint8 = disambig_data.astype(np.uint8) if disambig_data.dtype != np.uint8 else disambig_data
        flip_mask = (disambig_uint8 & 0x01) == 1  # Bit 0 is set
        
        # Add 180 degrees (π radians) to azimuth where flip_mask is True
        chi_corrected = chi.copy()
        chi_corrected[flip_mask] = (chi_corrected[flip_mask] + np.pi) % (2 * np.pi)
        chi = chi_corrected
        print(f"Applied disambiguation: {np.sum(flip_mask)} pixels flipped (out of {H*W} total)")
    
    # Convert spherical coordinates to Cartesian (Bx, By, Bz)
    Bx = B_mag_data * np.sin(theta) * np.cos(chi)
    By = B_mag_data * np.sin(theta) * np.sin(chi)
    Bz = B_mag_data * np.cos(theta)
    
    # Stack into [3, H, W] array
    magnetogram_obs = np.stack([Bx, By, Bz], axis=0)
    
    return magnetogram_obs, H, W


def plot_magnetogram_comparison(magnetogram_surya, magnetogram_obs, timestamp=None, save_path=None):
    """
    Plot Surya-generated and observed magnetograms side by side for comparison.
    
    Args:
        magnetogram_surya: Numpy array of shape [3, H, W] with Surya-generated Bx, By, Bz
        magnetogram_obs: Numpy array of shape [3, H, W] with observed Bx, By, Bz
        timestamp: Timestamp string for title (optional)
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    component_names = ['Bx', 'By', 'Bz']
    
    # Downsample if images are too large for display
    H_surya, W_surya = magnetogram_surya.shape[1], magnetogram_surya.shape[2]
    H_obs, W_obs = magnetogram_obs.shape[1], magnetogram_obs.shape[2]
    
    # Downsample to 512x512 for display if larger
    max_display_size = 512
    if H_surya > max_display_size or W_surya > max_display_size:
        # Ensure array is contiguous before converting to tensor
        magnetogram_surya_contig = np.ascontiguousarray(magnetogram_surya)
        mag_surya_tensor = torch.from_numpy(magnetogram_surya_contig).float().unsqueeze(0)  # [1, 3, H, W]
        mag_surya_display_tensor = F.interpolate(mag_surya_tensor, size=(max_display_size, max_display_size), 
                                                 mode='bilinear', align_corners=False)
        magnetogram_surya_display = mag_surya_display_tensor[0].numpy()  # [3, H, W]
    else:
        magnetogram_surya_display = magnetogram_surya
    
    if H_obs > max_display_size or W_obs > max_display_size:
        # Ensure array is contiguous before converting to tensor
        magnetogram_obs_contig = np.ascontiguousarray(magnetogram_obs)
        mag_obs_tensor = torch.from_numpy(magnetogram_obs_contig).float().unsqueeze(0)  # [1, 3, H, W]
        mag_obs_display_tensor = F.interpolate(mag_obs_tensor, size=(max_display_size, max_display_size),
                                               mode='bilinear', align_corners=False)
        magnetogram_obs_display = mag_obs_display_tensor[0].numpy()  # [3, H, W]
    else:
        magnetogram_obs_display = magnetogram_obs
    
    # Plot each component (Bx, By, Bz)
    lim = 1000  # Fixed limit for all magnetogram components
    
    for i, comp_name in enumerate(component_names):
        # Surya magnetogram
        vmin_surya = magnetogram_surya_display[i].min()
        vmax_surya = magnetogram_surya_display[i].max()
        im1 = axes[i, 0].imshow(magnetogram_surya_display[i], cmap='hmimag', origin='lower', 
                               vmin=-lim, vmax=lim)
        axes[i, 0].set_title(f'Surya {comp_name} (G)', fontsize=11)
        axes[i, 0].axis('off')
        plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Observed magnetogram
        vmin_obs = magnetogram_obs_display[i].min()
        vmax_obs = magnetogram_obs_display[i].max()
        im2 = axes[i, 1].imshow(magnetogram_obs_display[i], cmap=f'hmimag', origin='lower',
                               vmin=-lim, vmax=lim)
        axes[i, 1].set_title(f'Observed {comp_name} (G)', fontsize=11)
        axes[i, 1].axis('off')
        plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Difference (Surya - Observed)
        # Need to match sizes if different
        if magnetogram_surya_display.shape[1:] != magnetogram_obs_display.shape[1:]:
            # Interpolate observed to match Surya size using torch
            obs_comp_tensor = torch.from_numpy(magnetogram_obs_display[i]).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            target_h, target_w = magnetogram_surya_display.shape[1], magnetogram_surya_display.shape[2]
            obs_resized_tensor = F.interpolate(obs_comp_tensor, size=(target_h, target_w),
                                              mode='bilinear', align_corners=False)
            obs_resized = obs_resized_tensor[0, 0].numpy()
            diff = magnetogram_surya_display[i] - obs_resized
        else:
            diff = magnetogram_surya_display[i] - magnetogram_obs_display[i]
        
        vmin_diff = diff.min()
        vmax_diff = diff.max()
        im3 = axes[i, 2].imshow(diff, cmap=f'hmimag', origin='lower',
                               vmin=-lim, vmax=lim)
        axes[i, 2].set_title(f'Difference {comp_name} (G)', fontsize=11, fontweight='bold')
        axes[i, 2].axis('off')
        plt.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    # Add overall title
    title = 'Magnetogram Comparison: Surya vs Observed'
    if timestamp is not None:
        title += f'\n{timestamp}'
    fig.suptitle(title, fontsize=14, y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()
