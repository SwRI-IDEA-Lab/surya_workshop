"""
Inference script for applying pre-trained Stokes profile model to training set.

This script:
1. Loads the trained model from checkpoint
2. Applies it to the training dataset
3. Evaluates the predictions
4. Converts predicted I and V profiles to line-of-sight magnetogram
"""

import os
import sys
import torch
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add paths for imports
sys.path.append("../../")
sys.path.append("../../Surya")

from surya.utils.data import build_scalers
from datasets.stokes_profile_dataset import StokesProfileDataset
from models.stokes_baseline import StokesBaselineModel
from lightning_modules.pl_stokes_baseline import StokesLightningModule
from metrics.stokes_metrics import StokesMetrics


def load_trained_model(checkpoint_path, n_wavelengths=50, hidden_dim=128, learning_rate=1e-3, batch_size=1):
    """
    Load the trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        n_wavelengths: Number of wavelength points (must match training)
        hidden_dim: Hidden dimension (must match training)
        learning_rate: Learning rate (must match training)
        batch_size: Batch size (must match training)
    
    Returns:
        Trained model in evaluation mode
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Recreate model architecture
    model = StokesBaselineModel(
        n_wavelengths=n_wavelengths,
        hidden_dim=hidden_dim,
        use_conv=True,
    )
    
    # Recreate metrics
    metrics = {
        'train_loss': StokesMetrics("train_loss"),
        'train_metrics': StokesMetrics("train_metrics"),
        'val_metrics': StokesMetrics("val_metrics")
    }
    
    # Load from checkpoint
    trained_model = StokesLightningModule.load_from_checkpoint(
        checkpoint_path,
        model=model,
        metrics=metrics,
        lr=learning_rate,
        batch_size=batch_size
    )
    
    # Set to evaluation mode
    trained_model.eval()
    trained_model.freeze()
    
    print("✅ Model loaded successfully!")
    return trained_model


def stokes_to_los_magnetogram(stokes_I, stokes_V, wavelengths, lambda_rest=6173.15, geff=2.5):
    """
    Convert Stokes I and V profiles to line-of-sight (LOS) magnetogram using integration method.
    
    This uses a simple integration approach based on the weak field approximation.
    The method integrates the V profile relative to I to estimate the LOS magnetic field.
    
    Formula (simplified): B_los ≈ C * integral(V dλ) / integral(I dλ)
    where C is a calibration constant.
    
    Args:
        stokes_I: Stokes I profile [H, W, n_wavelengths] or [n_wavelengths, H, W]
        stokes_V: Stokes V profile [H, W, n_wavelengths] or [n_wavelengths, H, W]
        wavelengths: Wavelength array [n_wavelengths]
        lambda_rest: Rest wavelength in Angstroms (default: 6173.15 for HMI)
        geff: Effective Landé factor (default: 2.5 for HMI)
    
    Returns:
        los_magnetogram: Line-of-sight magnetic field in Gauss [H, W]
    """
    # Handle different input shapes
    if stokes_I.ndim == 3:
        if stokes_I.shape[0] == len(wavelengths):
            # Shape: [n_wavelengths, H, W]
            stokes_I = stokes_I.transpose(1, 2, 0)  # [H, W, n_wavelengths]
            stokes_V = stokes_V.transpose(1, 2, 0)  # [H, W, n_wavelengths]
        # Now shape is [H, W, n_wavelengths]
    
    H, W, n_wavelengths = stokes_I.shape
    
    # Convert to numpy if tensor
    if isinstance(stokes_I, torch.Tensor):
        stokes_I = stokes_I.cpu().numpy()
    if isinstance(stokes_V, torch.Tensor):
        stokes_V = stokes_V.cpu().numpy()
    if isinstance(wavelengths, torch.Tensor):
        wavelengths = wavelengths.cpu().numpy()
    
    # Initialize output
    los_magnetogram = np.zeros((H, W), dtype=np.float32)
    
    # Calibration constant (empirically determined, may need adjustment)
    # This is a simplified calibration - actual HMI uses more sophisticated methods
    # Typical value: ~1e5 to convert to Gauss
    calibration_constant = 1e5
    
    # Process each pixel
    for h in range(H):
        for w in range(W):
            I_profile = stokes_I[h, w, :]  # [n_wavelengths]
            V_profile = stokes_V[h, w, :]  # [n_wavelengths]
            
            # Method: Integrate V relative to I
            # This is based on the weak field approximation where V is proportional to B_los * dI/dλ
            area_V = np.trapz(V_profile, wavelengths)
            area_I = np.trapz(I_profile, wavelengths)
            
            # Alternative: Use peak-to-peak amplitude of V relative to I
            # This can be more robust for noisy data
            V_amplitude = np.max(V_profile) - np.min(V_profile)
            I_continuum = np.mean(I_profile)  # Approximate continuum level
            
            # Use area method if I is well-defined, otherwise use amplitude method
            if abs(area_I) > 1e-10:
                # Area-based method
                los_magnetogram[h, w] = (area_V / area_I) * calibration_constant
            elif I_continuum > 1e-10:
                # Amplitude-based method (fallback)
                los_magnetogram[h, w] = (V_amplitude / I_continuum) * calibration_constant * 0.1
            else:
                los_magnetogram[h, w] = 0.0
    
    return los_magnetogram


def run_inference_on_training_set(
    checkpoint_path,
    config_path,
    hmi_b_dir,
    output_dir="./inference_results",
    max_samples=None,
    device='cuda'
):
    """
    Run inference on the training set and convert to LOS magnetogram.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config.yaml
        hmi_b_dir: Directory containing HMI B FITS files
        output_dir: Directory to save results
        max_samples: Maximum number of samples to process (None for all)
        device: Device to run inference on
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Stokes Profile Inference on Training Set")
    print("=" * 60)
    
    # Load configuration
    print("\n📋 Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config["data"]["scalers"] = yaml.safe_load(open(config["data"]["scalers_path"], "r"))
    scalers = build_scalers(info=config["data"]["scalers"])
    
    # Create dataset
    print("\n📊 Creating training dataset...")
    wavelengths = np.linspace(6172.65, 6173.65, 50)
    
    train_dataset = StokesProfileDataset(
        index_path=config["data"]["train_data_path"],
        hmi_b_dir=hmi_b_dir,
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        n_input_timestamps=config["model"]["time_embedding"]["time_dim"],
        rollout_steps=config["rollout_steps"],
        channels=config["data"]["channels"],
        drop_hmi_probability=config["drop_hmi_probability"],
        use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
        scalers=scalers,
        phase="train",
        s3_use_simplecache=True,
        s3_cache_dir="/tmp/helio_s3_cache",
        wavelengths=wavelengths,
        pixel_batch_size=10000,
        device='cpu',  # Use CPU for synthesis
        max_number_of_samples=max_samples,
    )
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    # Load trained model
    print("\n🤖 Loading trained model...")
    trained_model = load_trained_model(
        checkpoint_path,
        n_wavelengths=len(wavelengths),
        hidden_dim=128,
        learning_rate=1e-3,
        batch_size=1
    )
    trained_model = trained_model.to(device)
    
    # Initialize metrics
    val_metrics = StokesMetrics("val_metrics")
    
    # Storage for results
    all_predictions = []
    all_targets = []
    all_inputs = []
    all_los_pred = []
    all_los_target = []
    all_metrics = []
    
    print("\n🔬 Running inference on training set...")
    print(f"   Dataset size: {len(train_dataset)}")
    print(f"   Device: {device}")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
        # Move batch to device
        batch['stokes_input'] = batch['stokes_input'].to(device)
        batch['forecast'] = batch['forecast'].to(device)
        
        # Run inference
        with torch.no_grad():
            predictions = trained_model(batch)  # [B, 4, n_wavelengths, H, W]
        
        # Move back to CPU for processing
        predictions = predictions.cpu()
        targets = batch['forecast'].cpu()
        inputs = batch['stokes_input'].cpu()
        
        # Compute metrics
        metrics_dict, _ = val_metrics(predictions, targets)
        all_metrics.append({k: v.item() for k, v in metrics_dict.items()})
        
        # Extract I and V profiles (indices 0=I, 3=V)
        pred_I = predictions[0, 0, :, :, :].numpy()  # [n_wavelengths, H, W]
        pred_V = predictions[0, 3, :, :, :].numpy()  # [n_wavelengths, H, W]
        target_I = targets[0, 0, :, :, :].numpy()  # [n_wavelengths, H, W]
        target_V = targets[0, 3, :, :, :].numpy()  # [n_wavelengths, H, W]
        
        # Convert to LOS magnetogram
        los_pred = stokes_to_los_magnetogram(pred_I, pred_V, wavelengths)
        los_target = stokes_to_los_magnetogram(target_I, target_V, wavelengths)
        
        # Store results
        all_predictions.append(predictions.numpy())
        all_targets.append(targets.numpy())
        all_inputs.append(inputs.numpy())
        all_los_pred.append(los_pred)
        all_los_target.append(los_target)
        
        # Save sample results for first few batches
        if batch_idx < 3:
            sample_dir = output_dir / f"sample_{batch_idx}"
            sample_dir.mkdir(exist_ok=True)
            
            # Save Stokes profiles
            np.save(sample_dir / "stokes_predicted.npy", predictions.numpy())
            np.save(sample_dir / "stokes_target.npy", targets.numpy())
            np.save(sample_dir / "stokes_input.npy", inputs.numpy())
            np.save(sample_dir / "wavelengths.npy", wavelengths)
            
            # Save LOS magnetograms
            np.save(sample_dir / "los_magnetogram_predicted.npy", los_pred)
            np.save(sample_dir / "los_magnetogram_target.npy", los_target)
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Plot LOS magnetograms
            vmin = min(los_pred.min(), los_target.min())
            vmax = max(los_pred.max(), los_target.max())
            
            im1 = axes[0, 0].imshow(los_pred, cmap='RdBu', vmin=vmin, vmax=vmax, origin='lower')
            axes[0, 0].set_title('Predicted LOS Magnetogram (G)')
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0])
            
            im2 = axes[0, 1].imshow(los_target, cmap='RdBu', vmin=vmin, vmax=vmax, origin='lower')
            axes[0, 1].set_title('Target LOS Magnetogram (G)')
            axes[0, 1].axis('off')
            plt.colorbar(im2, ax=axes[0, 1])
            
            im3 = axes[0, 2].imshow(los_pred - los_target, cmap='RdBu', origin='lower')
            axes[0, 2].set_title('Difference (Pred - Target)')
            axes[0, 2].axis('off')
            plt.colorbar(im3, ax=axes[0, 2])
            
            # Plot Stokes V profiles for a few pixels
            center_h, center_w = los_pred.shape[0] // 2, los_pred.shape[1] // 2
            pixels_to_plot = [
                (center_h, center_w, 'Center'),
                (center_h + 50, center_w, 'Offset 1'),
                (center_h, center_w + 50, 'Offset 2'),
            ]
            
            for idx, (h, w, label) in enumerate(pixels_to_plot):
                if h < pred_V.shape[1] and w < pred_V.shape[2]:
                    axes[1, idx].plot(wavelengths, pred_V[:, h, w], 'b-', label='Predicted V', linewidth=2)
                    axes[1, idx].plot(wavelengths, target_V[:, h, w], 'r--', label='Target V', linewidth=2)
                    axes[1, idx].set_xlabel('Wavelength (Å)')
                    axes[1, idx].set_ylabel('Stokes V')
                    axes[1, idx].set_title(f'Stokes V - {label}')
                    axes[1, idx].legend()
                    axes[1, idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(sample_dir / "visualization.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   Saved sample {batch_idx} results to {sample_dir}")
    
    # Compute summary statistics
    print("\n📊 Summary Statistics:")
    print("=" * 60)
    
    # Average metrics across all samples
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        print(f"  {key}: {avg_metrics[key]:.6f}")
    
    # LOS magnetogram statistics
    all_los_pred_array = np.array(all_los_pred)
    all_los_target_array = np.array(all_los_target)
    
    los_mse = np.mean((all_los_pred_array - all_los_target_array) ** 2)
    los_mae = np.mean(np.abs(all_los_pred_array - all_los_target_array))
    
    print(f"\n  LOS Magnetogram MSE: {los_mse:.6f}")
    print(f"  LOS Magnetogram MAE: {los_mae:.6f}")
    print(f"  LOS Magnetogram Pred Range: [{all_los_pred_array.min():.2f}, {all_los_pred_array.max():.2f}] G")
    print(f"  LOS Magnetogram Target Range: [{all_los_target_array.min():.2f}, {all_los_target_array.max():.2f}] G")
    
    # Save summary
    summary = {
        'n_samples': len(all_predictions),
        'metrics': avg_metrics,
        'los_magnetogram_mse': float(los_mse),
        'los_magnetogram_mae': float(los_mae),
    }
    
    import json
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Inference complete! Results saved to: {output_dir}")
    return summary, all_los_pred, all_los_target


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference on training set")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="./configs/config.yaml",
                       help="Path to config.yaml")
    parser.add_argument("--hmi_b_dir", type=str, default="./datasets/hmi.B",
                       help="Directory containing HMI B FITS files")
    parser.add_argument("--output_dir", type=str, default="./inference_results",
                       help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    run_inference_on_training_set(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        hmi_b_dir=args.hmi_b_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        device=args.device
    )
