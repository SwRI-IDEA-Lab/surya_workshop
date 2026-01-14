"""
Utility functions for magnetogram testing and Stokes profile synthesis.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import sys

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
    
    return magnetogram_np, velocity_np, dataset


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
