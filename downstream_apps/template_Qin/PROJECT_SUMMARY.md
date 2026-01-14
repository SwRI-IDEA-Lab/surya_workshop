# Surya Magnetogram Radiative Transfer Testing Project

## Summary

This project provides a complete framework for testing Surya-generated magnetograms against the radiative transfer equation using the Milne-Eddington (ME) approximation. The implementation is specifically adapted for HMI Fe I 6173.15 Å line with effective Landé factor g_eff = 2.5.

## What Was Created

### 1. ME PINN Module (`ME_PINN_legacy/me_pinn_hmi.py`)

**Key Features:**
- Adapted from original legacy code for HMI line parameters
- **Wavelength**: 6173.15 Å (changed from 15648.5 Å)
- **Effective Landé factor**: 2.5 (changed from 3.0)
- Implements full ME forward model with Voigt profiles
- Includes adaptive loss weighting for Stokes components

**Classes:**
- `MEInversionPINN`: Neural network for ME parameter prediction
- `MEPhysicsLoss`: Physics-informed loss using ME forward model
- `METotalLoss`: Combined loss with adaptive weighting

### 2. Magnetogram Dataset (`datasets/magnetogram_dataset.py`)

**Key Features:**
- Extends `HelioNetCDFDatasetAWS` for magnetogram-specific functionality
- Extracts magnetogram components (Bx, By, Bz) from Surya output
- Optional support for loading observed Stokes profiles
- Compatible with Surya data pipeline

### 3. Testing Script (`test_magnetogram_rt.py`)

**Key Features:**
- Loads Surya-generated magnetograms
- Converts magnetograms to Stokes profiles using ME forward model
- Validates consistency by recomputing Stokes from ME parameters
- Generates comprehensive reports with statistics and visualizations

**Functions:**
- `compute_magnetogram_to_stokes()`: Converts (Bx, By, Bz) to Stokes (I, Q, U, V)
- `test_magnetogram_rt()`: Main testing workflow

### 4. Configuration Files

- `configs/config_magnetogram_test.yaml`: Configuration for magnetogram testing
- Includes ME PINN parameters, testing options, and Surya model settings

### 5. Documentation

- `ME_PINN_legacy/README.md`: Detailed ME PINN documentation
- `README_MAGNETOGRAM_TEST.md`: Project overview and usage guide
- `PROJECT_SUMMARY.md`: This file

## Key Changes from Legacy Code

### Line Parameters
- **Original**: λ = 15648.5 Å, g_eff = 3.0
- **New (HMI)**: λ = 6173.15 Å, g_eff = 2.5

### Code Structure
- Separated physics model from training code
- Added vectorized batch processing
- Improved memory efficiency
- Better integration with Surya pipeline

## Usage Examples

### Basic Testing

```bash
cd downstream_apps/template_Qin
conda activate surya_ws
python test_magnetogram_rt.py \
    --config configs/config_magnetogram_test.yaml \
    --n_samples 10 \
    --device cuda
```

### Using ME PINN Directly

```python
from ME_PINN_legacy.me_pinn_hmi import MEPhysicsLoss
import torch
import numpy as np

# Initialize for HMI
physics_loss_fn = MEPhysicsLoss(lambda_rest=6173.15, geff=2.5)

# Synthesize Stokes profiles
wavelengths = np.linspace(6172.65, 6173.65, 50)
wavelengths_tensor = torch.tensor(wavelengths, dtype=torch.float32)

# ME parameters: [B, theta, chi, eta0, dlambdaD, a, lambda0, B0, B1]
params = torch.tensor([[1000.0, 0.5, 0.3, 2.0, 0.15, 0.1, 0.0, 1.0, 0.5]])

_, stokes = physics_loss_fn(params, wavelengths_tensor, None)
```

### Converting Magnetogram to Stokes

```python
from test_magnetogram_rt import compute_magnetogram_to_stokes

# magnetogram: [B, 3, H, W] with Bx, By, Bz
stokes, me_params = compute_magnetogram_to_stokes(
    magnetogram, wavelengths, device='cuda'
)
```

## Project Structure

```
template_Qin/
├── ME_PINN_legacy/
│   ├── __init__.py
│   ├── me_pinn_hmi.py          # HMI-adapted ME PINN
│   ├── legacy_training.py      # Original code (reference)
│   └── README.md
├── datasets/
│   ├── magnetogram_dataset.py  # Magnetogram dataset class
│   └── template_dataset.py
├── configs/
│   ├── config.yaml
│   └── config_magnetogram_test.yaml
├── test_magnetogram_rt.py      # Main testing script
├── README_MAGNETOGRAM_TEST.md  # Usage guide
└── PROJECT_SUMMARY.md          # This file
```

## Next Steps

### For Fine-tuning Surya

1. Use existing fine-tuning templates (`2_finetune_template_1D.ipynb`)
2. Modify loss function to include RT consistency terms
3. Use ME PINN as physics-informed regularization

### For Validation

1. Run `test_magnetogram_rt.py` on generated magnetograms
2. Analyze RT consistency errors
3. Compare synthesized vs observed Stokes profiles (if available)

### For Development

1. Adjust ME parameters (eta0, dlambdaD, etc.) based on observations
2. Extend to support multiple spectral lines
3. Add comparison with observed Stokes profiles

## Technical Details

### ME Model Parameters

The model predicts 9 parameters:
1. **B**: Magnetic field strength [0, 4500] G
2. **theta**: Inclination [0, π]
3. **chi**: Azimuth [0, π]
4. **eta0**: Line-to-continuum opacity [0.5, 20]
5. **dlambdaD**: Doppler width [0.12, 0.25] Å
6. **a**: Damping [0, 10]
7. **lambda0**: Line center shift [-0.25, 0.25] Å
8. **B0**: Continuum source function
9. **B1**: Line source function

### Conversion: Magnetogram → ME Parameters

- **B** = √(Bx² + By² + Bz²)
- **theta** = arccos(Bz / B)
- **chi** = arctan2(By, Bx) normalized to [0, π]

### Physics Model

The ME forward model:
1. Calculates Zeeman splitting: vb = g_eff × (4.67e-13 × λ² × B) / dlambdaD
2. Computes Voigt profiles for blue, center, and red components
3. Synthesizes Stokes I, Q, U, V profiles

## Dependencies

- PyTorch
- Surya foundation model
- NumPy, Matplotlib
- Standard scientific Python stack

## Environment

```bash
conda activate surya_ws
```

## References

- Surya Foundation Model: [GitHub](https://github.com/NASA-IMPACT/Surya)
- Milne-Eddington approximation
- HMI Fe I 6173.15 Å line parameters

## Notes

- The original legacy code is preserved for reference
- All new code is adapted for HMI line parameters
- The implementation is optimized for batch processing
- Memory-efficient vectorization for large magnetograms
