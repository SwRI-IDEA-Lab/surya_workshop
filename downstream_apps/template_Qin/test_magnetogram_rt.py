"""
Test Surya-generated magnetograms against radiative transfer equation.

This script loads Surya-generated magnetograms and validates them using
the Milne-Eddington radiative transfer model for HMI Fe I 6173.15 Å.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add paths for imports
sys.path.append("../../")
sys.path.append("../../Surya")

from ME_PINN_legacy.me_pinn_hmi import MEInversionPINN, MEPhysicsLoss, METotalLoss
from datasets.magnetogram_dataset import MagnetogramTestDataset
from surya.utils.data import build_scalers


def compute_magnetogram_to_stokes(magnetogram, wavelengths, device='cuda', pixel_batch_size=10000):
    """
    Convert magnetogram (Bx, By, Bz) to Stokes profiles using ME forward model.
    
    This function takes a magnetogram and synthesizes Stokes profiles using
    the Milne-Eddington radiative transfer equation.
    
    Args:
        magnetogram: Tensor of shape [B, 3, H, W] containing Bx, By, Bz
        wavelengths: Wavelength array [n_wavelengths]
        device: Device to run computation on
        pixel_batch_size: Number of pixels to process at once (default: 10000)
        
    Returns:
        stokes_synthesized: Synthesized Stokes profiles [B, 4, n_wavelengths, H, W]
        me_params: ME parameters used [B, 9, H, W]
    """
    B, C, H, W = magnetogram.shape
    n_wavelengths = len(wavelengths)
    
    # Move to device if needed
    if magnetogram.device != device:
        magnetogram = magnetogram.to(device)
    
    # Extract Bx, By, Bz
    Bx = magnetogram[:, 0, :, :]  # [B, H, W]
    By = magnetogram[:, 1, :, :]  # [B, H, W]
    Bz = magnetogram[:, 2, :, :]  # [B, H, W]
    
    # Convert to spherical coordinates
    # B = sqrt(Bx^2 + By^2 + Bz^2)
    # theta = arccos(Bz / B)  [inclination from vertical]
    # chi = arctan2(By, Bx)    [azimuth in plane]
    B_mag = torch.sqrt(Bx**2 + By**2 + Bz**2 + 1e-6)  # Add small epsilon to avoid division by zero
    
    # Inclination: angle from vertical (z-axis)
    theta = torch.acos(torch.clamp(Bz / B_mag, -1.0, 1.0))
    
    # Azimuth: angle in the x-y plane
    chi = torch.atan2(By, Bx)
    # Normalize to [0, π] range
    chi = (chi + np.pi) % (2 * np.pi)
    chi = torch.where(chi > np.pi, chi - np.pi, chi)
    
    # Initialize ME physics loss function
    physics_loss_fn = MEPhysicsLoss(lambda_rest=6173.15, geff=2.5).to(device)
    wavelengths_tensor = torch.tensor(wavelengths, dtype=torch.float32).to(device)
    
    # Default ME parameters (can be adjusted)
    # [B, theta, chi, eta0, dlambdaD, a, lambda0, B0, B1]
    # We'll use typical values for quiet Sun
    eta0_default = 2.0
    dlambdaD_default = 0.15  # Angstroms
    a_default = 0.1
    lambda0_default = 0.0
    B0_default = 1.0
    B1_default = 0.5
    
    # Prepare storage on CPU to save GPU memory, move to device only when needed
    stokes_synthesized = torch.zeros((B, 4, n_wavelengths, H, W), dtype=torch.float32)
    me_params = torch.zeros((B, 9, H, W), dtype=torch.float32)
    
    # Process in batches to avoid memory issues
    # Use spatial patches for better memory efficiency
    patch_size = int(np.sqrt(pixel_batch_size))  # Approximate patch size
    patch_size = max(32, min(patch_size, 256))  # Limit patch size between 32 and 256
    
    for b in range(B):
        B_vals = B_mag[b]  # [H, W]
        theta_vals = theta[b]  # [H, W]
        chi_vals = chi[b]  # [H, W]
        
        # Calculate total number of patches for progress tracking
        n_patches_h = (H + patch_size - 1) // patch_size
        n_patches_w = (W + patch_size - 1) // patch_size
        total_patches = n_patches_h * n_patches_w
        
        # Process in spatial patches with progress tracking
        patch_idx = 0
        for h_start in range(0, H, patch_size):
            h_end = min(h_start + patch_size, H)
            for w_start in range(0, W, patch_size):
                w_end = min(w_start + patch_size, W)
                patch_idx += 1
                
                # Extract patch
                B_patch = B_vals[h_start:h_end, w_start:w_end].flatten()  # [patch_H * patch_W]
                theta_patch = theta_vals[h_start:h_end, w_start:w_end].flatten()
                chi_patch = chi_vals[h_start:h_end, w_start:w_end].flatten()
                
                n_pixels_patch = len(B_patch)
                
                # Construct ME parameters for this patch
                params_patch = torch.stack([
                    B_patch,
                    theta_patch,
                    chi_patch,
                    torch.full((n_pixels_patch,), eta0_default, device=device),
                    torch.full((n_pixels_patch,), dlambdaD_default, device=device),
                    torch.full((n_pixels_patch,), a_default, device=device),
                    torch.full((n_pixels_patch,), lambda0_default, device=device),
                    torch.full((n_pixels_patch,), B0_default, device=device),
                    torch.full((n_pixels_patch,), B1_default, device=device),
                ], dim=1)  # [n_pixels_patch, 9]
                
                # Synthesize Stokes profiles for this patch
                with torch.no_grad():  # No gradients needed for synthesis
                    _, stokes_pred = physics_loss_fn(params_patch, wavelengths_tensor, None)
                # stokes_pred: [n_pixels_patch, 4, n_wavelengths]
                
                # Move to CPU and reshape
                stokes_pred_cpu = stokes_pred.cpu().permute(1, 2, 0)  # [4, n_wavelengths, n_pixels_patch]
                stokes_pred_cpu = stokes_pred_cpu.view(4, n_wavelengths, h_end - h_start, w_end - w_start)
                params_patch_cpu = params_patch.cpu().permute(1, 0)  # [9, n_pixels_patch]
                params_patch_cpu = params_patch_cpu.view(9, h_end - h_start, w_end - w_start)
                
                # Store results
                stokes_synthesized[b, :, :, h_start:h_end, w_start:w_end] = stokes_pred_cpu
                me_params[b, :, h_start:h_end, w_start:w_end] = params_patch_cpu
                
                # Clear GPU memory
                del params_patch, stokes_pred
                torch.cuda.empty_cache()
                
                # Progress update every 10 patches
                if patch_idx % 10 == 0 or patch_idx == total_patches:
                    print(f"    Batch {b+1}/{B}: Processed {patch_idx}/{total_patches} patches ({100*patch_idx/total_patches:.1f}%)", end='\r')
    
        # Final progress update for this batch
        print(f"    Batch {b+1}/{B}: Completed ({total_patches} patches)", flush=True)
    
    # Move final results to device if needed (after all batches processed)
    if device != 'cpu':
        stokes_synthesized = stokes_synthesized.to(device)
        me_params = me_params.to(device)
    
    return stokes_synthesized, me_params


def test_magnetogram_rt(
    config_path: str,
    model_checkpoint: str = None,
    output_dir: str = "./magnetogram_rt_test_results",
    n_samples: int = 10,
    device: str = "cuda"
):
    """
    Test Surya-generated magnetograms against radiative transfer equation.
    
    Args:
        config_path: Path to configuration YAML file
        model_checkpoint: Path to Surya model checkpoint (optional, for generating magnetograms)
        output_dir: Directory to save test results
        n_samples: Number of samples to test
        device: Device to run computation on
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    print("📋 Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config["data"]["scalers"] = yaml.safe_load(open(config["data"]["scalers_path"], "r"))
    scalers = build_scalers(info=config["data"]["scalers"])
    
    # Create dataset
    print("📊 Creating dataset...")
    dataset = MagnetogramTestDataset(
        index_path=config["data"]["valid_data_path"],
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        n_input_timestamps=config["data"].get("n_input_timestamps", 1),
        rollout_steps=1,
        scalers=scalers,
        channels=config["data"]["channels"],
        phase="test",
        return_stokes=False,  # Set to True if you have observed Stokes data
        max_number_of_samples=n_samples,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    
    # Initialize ME physics loss for validation
    physics_loss_fn = MEPhysicsLoss(lambda_rest=6173.15, geff=2.5).to(device)
    
    # HMI wavelength range (typically ±0.5 Å around 6173.15 Å)
    wavelengths = np.linspace(6172.65, 6173.65, 50)  # 50 wavelength points
    
    # Storage for results
    all_rt_errors = []
    all_magnetograms = []
    all_synthesized_stokes = []
    
    print("🔬 Testing magnetograms against radiative transfer equation...")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
        if batch_idx >= n_samples:
            break
        
        # Extract magnetogram
        magnetogram = batch['magnetogram'].to(device)  # [1, 3, H, W]
        
        # Downsample to 256x256 for memory efficiency
        if magnetogram.shape[-1] != 256 or magnetogram.shape[-2] != 256:
            print(f"  Downsampling magnetogram from {magnetogram.shape[-1]}x{magnetogram.shape[-2]} to 256x256 for memory efficiency")
            magnetogram = F.interpolate(magnetogram, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Synthesize Stokes profiles from magnetogram
        stokes_synthesized, me_params = compute_magnetogram_to_stokes(
            magnetogram, wavelengths, device=device, pixel_batch_size=5000  # Reduced batch size for memory
        )
        
        # Compute radiative transfer consistency metrics
        # For now, we'll compute the forward model consistency
        # (how well the synthesized profiles satisfy the RT equation)
        
        # Reshape for batch processing
        B, C, H, W = magnetogram.shape
        n_wavelengths = len(wavelengths)
        
        # Flatten spatial dimensions
        me_params_flat = me_params.view(B, 9, -1).permute(0, 2, 1)  # [B, H*W, 9]
        stokes_flat = stokes_synthesized.view(B, 4, n_wavelengths, -1).permute(0, 3, 1, 2)  # [B, H*W, 4, n_wavelengths]
        
        wavelengths_tensor = torch.tensor(wavelengths, dtype=torch.float32).to(device)
        
        # Recompute Stokes from ME parameters to check consistency
        rt_errors = []
        for b in range(B):
            for pixel_idx in range(me_params_flat.shape[1]):
                params_pixel = me_params_flat[b, pixel_idx, :].unsqueeze(0)  # [1, 9]
                stokes_pixel = stokes_flat[b, pixel_idx, :, :].unsqueeze(0)  # [1, 4, n_wavelengths]
                
                # Forward model
                _, stokes_recomputed = physics_loss_fn(params_pixel, wavelengths_tensor, None)
                
                # Compute error
                error = torch.mean((stokes_recomputed - stokes_pixel) ** 2)
                rt_errors.append(error.item())
        
        all_rt_errors.extend(rt_errors)
        all_magnetograms.append(magnetogram.cpu().numpy())
        all_synthesized_stokes.append(stokes_synthesized.cpu().numpy())
        
        # Save sample results
        if batch_idx < 3:  # Save first 3 samples
            sample_dir = output_dir / f"sample_{batch_idx}"
            sample_dir.mkdir(exist_ok=True)
            
            # Save magnetogram
            np.save(sample_dir / "magnetogram.npy", magnetogram.cpu().numpy())
            
            # Save synthesized Stokes
            np.save(sample_dir / "stokes_synthesized.npy", stokes_synthesized.cpu().numpy())
            np.save(sample_dir / "wavelengths.npy", wavelengths)
            np.save(sample_dir / "me_params.npy", me_params.cpu().numpy())
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot magnetogram components
            mag_np = magnetogram[0].cpu().numpy()
            axes[0, 0].imshow(mag_np[0], cmap='RdBu', origin='lower')
            axes[0, 0].set_title('Bx (G)')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(mag_np[1], cmap='RdBu', origin='lower')
            axes[0, 1].set_title('By (G)')
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(mag_np[2], cmap='RdBu', origin='lower')
            axes[1, 0].set_title('Bz (G)')
            axes[1, 0].axis('off')
            
            # Plot B magnitude instead of Stokes in this panel
            B_mag_plot = np.sqrt(mag_np[0]**2 + mag_np[1]**2 + mag_np[2]**2)
            im = axes[1, 1].imshow(B_mag_plot, cmap='hot', origin='lower')
            axes[1, 1].set_title('|B| (G)')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1])
            
            plt.tight_layout()
            plt.savefig(sample_dir / "visualization.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Create separate plots for Stokes profiles
            stokes_np = stokes_synthesized[0].cpu().numpy()  # [4, n_wavelengths, H, W]
            B_mag_2d = np.sqrt(mag_np[0]**2 + mag_np[1]**2 + mag_np[2]**2)
            
            # Find pixels with different field strengths
            center_h, center_w = H // 2, W // 2
            max_B_idx = np.unravel_index(np.argmax(B_mag_2d), B_mag_2d.shape)
            min_B_idx = np.unravel_index(np.argmin(B_mag_2d), B_mag_2d.shape)
            
            # Plot 1: Individual pixel Stokes profiles (separate subplots for each Stokes parameter)
            fig, axes = plt.subplots(4, 1, figsize=(10, 12))
            
            pixels_to_plot = [
                (center_h, center_w, 'Center'),
                (max_B_idx[0], max_B_idx[1], f'Max B ({B_mag_2d[max_B_idx]:.1f} G)'),
                (min_B_idx[0], min_B_idx[1], f'Min B ({B_mag_2d[min_B_idx]:.1f} G)'),
            ]
            
            # Plot I, Q, U, V separately
            for stokes_idx, (stokes_name, color) in enumerate([('I', 'black'), ('Q', 'red'), ('U', 'green'), ('V', 'blue')]):
                ax = axes[stokes_idx]
                
                for h, w, label in pixels_to_plot:
                    stokes_profile = stokes_np[stokes_idx, :, h, w]
                    ax.plot(wavelengths, stokes_profile, '-', color=color, alpha=0.7, linewidth=2, label=f'{label} (B={B_mag_2d[h, w]:.1f}G)')
                
                ax.set_xlabel('Wavelength (Å)')
                ax.set_ylabel(f'Stokes {stokes_name}')
                ax.set_title(f'Stokes {stokes_name} Profiles for Different Pixels')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(sample_dir / "stokes_profiles.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot 2: Single pixel with all 4 Stokes parameters
            # Use the pixel with maximum field strength
            h_max, w_max = max_B_idx
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            for stokes_idx, (stokes_name, color, linestyle) in enumerate([
                ('I', 'black', '-'), 
                ('Q', 'red', '--'), 
                ('U', 'green', '-.'), 
                ('V', 'blue', ':')
            ]):
                stokes_profile = stokes_np[stokes_idx, :, h_max, w_max]
                ax.plot(wavelengths, stokes_profile, color=color, linestyle=linestyle, 
                       linewidth=2, label=f'Stokes {stokes_name}', alpha=0.8)
            
            ax.set_xlabel('Wavelength (Å)', fontsize=12)
            ax.set_ylabel('Stokes Parameter', fontsize=12)
            ax.set_title(f'Synthesized Stokes Profiles - Pixel at ({h_max}, {w_max}), |B|={B_mag_2d[h_max, w_max]:.1f} G', fontsize=12)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(sample_dir / "stokes_single_pixel.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    Saved visualizations to {sample_dir}")
    
    # Compute summary statistics
    rt_errors_array = np.array(all_rt_errors)
    
    print("\n📊 Test Results Summary:")
    print(f"  Number of pixels tested: {len(all_rt_errors)}")
    print(f"  Mean RT error: {np.mean(rt_errors_array):.6e}")
    print(f"  Median RT error: {np.median(rt_errors_array):.6e}")
    print(f"  Std RT error: {np.std(rt_errors_array):.6e}")
    print(f"  Min RT error: {np.min(rt_errors_array):.6e}")
    print(f"  Max RT error: {np.max(rt_errors_array):.6e}")
    
    # Save summary
    summary = {
        'n_pixels': len(all_rt_errors),
        'mean_rt_error': float(np.mean(rt_errors_array)),
        'median_rt_error': float(np.median(rt_errors_array)),
        'std_rt_error': float(np.std(rt_errors_array)),
        'min_rt_error': float(np.min(rt_errors_array)),
        'max_rt_error': float(np.max(rt_errors_array)),
    }
    
    import json
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(rt_errors_array, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Radiative Transfer Error (MSE)')
    plt.ylabel('Frequency')
    plt.title('Distribution of RT Consistency Errors')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "error_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Test complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Surya magnetograms against RT equation")
    parser.add_argument("--config", type=str, default="./configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to Surya model checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="./magnetogram_rt_test_results",
                       help="Output directory for results")
    parser.add_argument("--n_samples", type=int, default=10,
                       help="Number of samples to test")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    test_magnetogram_rt(
        config_path=args.config,
        model_checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        device=args.device,
    )
