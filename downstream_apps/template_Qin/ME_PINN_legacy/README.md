# Milne-Eddington Inversion PINN for HMI Magnetograms

This directory contains Physics-Informed Neural Network (PINN) implementations for testing Surya-generated magnetograms against the radiative transfer equation.

## Overview

The Milne-Eddington (ME) approximation is a widely used model for solar magnetic field inversions. This implementation:

1. **Validates magnetograms**: Tests whether Surya-generated magnetograms satisfy the radiative transfer equation
2. **Synthesizes Stokes profiles**: Converts magnetograms (Bx, By, Bz) to Stokes profiles (I, Q, U, V) using the ME forward model
3. **Uses HMI line parameters**: Adapted for HMI Fe I 6173.15 Å with effective Landé factor g_eff = 2.5

## Files

- `me_pinn_hmi.py`: ME PINN implementation adapted for HMI line (6173.15 Å, g_eff=2.5)
- `legacy_training.py`: Original ME PINN training code (for reference, uses different line parameters)

## Key Parameters

### HMI Line Parameters
- **Wavelength**: 6173.15 Å (Fe I line)
- **Effective Landé factor (g_eff)**: 2.5
- **Wavelength range**: Typically ±0.5 Å around line center

### ME Model Parameters
The model predicts 9 parameters:
1. **B**: Magnetic field strength [0, 4500] G
2. **theta**: Inclination angle [0, π]
3. **chi**: Azimuth angle [0, π]
4. **eta0**: Line-to-continuum opacity ratio [0.5, 20]
5. **dlambdaD**: Doppler width [0.12, 0.25] Å
6. **a**: Damping parameter [0, 10]
7. **lambda0**: Line center shift [-0.25, 0.25] Å
8. **B0**: Continuum source function
9. **B1**: Line source function

## Usage

### Testing Magnetograms

```python
from ME_PINN_legacy.me_pinn_hmi import MEPhysicsLoss
import torch
import numpy as np

# Initialize physics loss function
physics_loss_fn = MEPhysicsLoss(lambda_rest=6173.15, geff=2.5)

# Define wavelength array (HMI typically uses ±0.5 Å around 6173.15 Å)
wavelengths = np.linspace(6172.65, 6173.65, 50)
wavelengths_tensor = torch.tensor(wavelengths, dtype=torch.float32)

# ME parameters [B, theta, chi, eta0, dlambdaD, a, lambda0, B0, B1]
params = torch.tensor([[1000.0, 0.5, 0.3, 2.0, 0.15, 0.1, 0.0, 1.0, 0.5]])

# Synthesize Stokes profiles
_, stokes_pred = physics_loss_fn(params, wavelengths_tensor, None)
# stokes_pred shape: [batch_size, 4, n_wavelengths]
```

### Converting Magnetogram to Stokes

```python
from test_magnetogram_rt import compute_magnetogram_to_stokes

# magnetogram: [B, 3, H, W] containing Bx, By, Bz
# wavelengths: array of wavelength points
stokes_synthesized, me_params = compute_magnetogram_to_stokes(
    magnetogram, wavelengths, device='cuda'
)
```

## Physics Model

The ME forward model solves the radiative transfer equation under the Milne-Eddington approximation:

1. **Zeeman splitting**: Calculates the splitting of spectral lines due to magnetic field
2. **Voigt profiles**: Computes absorption and dispersion profiles using Voigt functions
3. **Stokes synthesis**: Generates I, Q, U, V profiles from ME parameters

The model uses the Faddeeva function (complex error function) for efficient Voigt profile computation.

## References

- Milne-Eddington approximation for solar magnetic field inversions
- HMI Fe I 6173.15 Å line parameters
- Physics-Informed Neural Networks (PINNs) for solar physics

## Notes

- The original `legacy_training.py` uses different line parameters (15648.5 Å, g_eff=3.0) for a different spectral line
- This implementation is specifically adapted for HMI vector magnetograms
- The model can be used for both validation and synthesis of Stokes profiles
