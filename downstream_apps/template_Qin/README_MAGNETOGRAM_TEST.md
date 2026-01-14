# Surya Magnetogram Radiative Transfer Testing

This project provides tools for fine-tuning the Surya foundation model to generate magnetograms and testing them against the radiative transfer equation using the Milne-Eddington (ME) approximation.

## Overview

The project consists of:

1. **ME PINN Module** (`ME_PINN_legacy/me_pinn_hmi.py`): Physics-Informed Neural Network implementation adapted for HMI Fe I 6173.15 Å line
2. **Magnetogram Dataset** (`datasets/magnetogram_dataset.py`): Dataset class for loading Surya-generated magnetograms
3. **Testing Script** (`test_magnetogram_rt.py`): Script to validate magnetograms against radiative transfer equation

## Setup

### Environment

Activate the conda environment:
```bash
conda activate surya_ws
```

### Dependencies

The project uses:
- PyTorch
- Surya foundation model
- Standard scientific Python stack (numpy, matplotlib, etc.)

## Quick Start

### 1. Test Existing Magnetograms

To test Surya-generated magnetograms against the radiative transfer equation:

```bash
cd downstream_apps/template_Qin
python test_magnetogram_rt.py \
    --config configs/config_magnetogram_test.yaml \
    --output_dir ./magnetogram_rt_test_results \
    --n_samples 10 \
    --device cuda
```

### 2. Using the ME PINN Module Directly

```python
from ME_PINN_legacy.me_pinn_hmi import MEPhysicsLoss
import torch
import numpy as np

# Initialize for HMI line (6173.15 Å, g_eff=2.5)
physics_loss_fn = MEPhysicsLoss(lambda_rest=6173.15, geff=2.5)

# Define wavelength array
wavelengths = np.linspace(6172.65, 6173.65, 50)  # ±0.5 Å around line center
wavelengths_tensor = torch.tensor(wavelengths, dtype=torch.float32)

# ME parameters: [B, theta, chi, eta0, dlambdaD, a, lambda0, B0, B1]
params = torch.tensor([[1000.0, 0.5, 0.3, 2.0, 0.15, 0.1, 0.0, 1.0, 0.5]])

# Synthesize Stokes profiles
_, stokes_pred = physics_loss_fn(params, wavelengths_tensor, None)
# stokes_pred: [batch_size, 4, n_wavelengths]
```

### 3. Convert Magnetogram to Stokes Profiles

```python
from test_magnetogram_rt import compute_magnetogram_to_stokes

# Load magnetogram: [B, 3, H, W] containing Bx, By, Bz
magnetogram = torch.tensor(...)  # Your magnetogram data
wavelengths = np.linspace(6172.65, 6173.65, 50)

# Synthesize Stokes profiles
stokes_synthesized, me_params = compute_magnetogram_to_stokes(
    magnetogram, wavelengths, device='cuda'
)
```

## Project Structure

```
template_Qin/
├── ME_PINN_legacy/
│   ├── me_pinn_hmi.py          # ME PINN for HMI line (6173.15 Å)
│   ├── legacy_training.py      # Original training code (reference)
│   └── README.md               # ME PINN documentation
├── datasets/
│   ├── magnetogram_dataset.py  # Dataset for magnetogram testing
│   └── template_dataset.py     # Base template dataset
├── configs/
│   ├── config.yaml             # Base configuration
│   └── config_magnetogram_test.yaml  # Magnetogram testing config
├── test_magnetogram_rt.py      # Main testing script
└── README_MAGNETOGRAM_TEST.md  # This file
```

## Key Features

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

## Testing Workflow

1. **Load Surya-generated magnetograms** from the dataset
2. **Extract magnetogram components** (Bx, By, Bz)
3. **Convert to ME parameters** (B, theta, chi from Bx, By, Bz)
4. **Synthesize Stokes profiles** using ME forward model
5. **Validate consistency** by recomputing Stokes from ME parameters
6. **Generate reports** with error statistics and visualizations

## Output

The testing script generates:

- **Summary statistics**: Mean, median, std of RT errors
- **Error distribution**: Histogram of RT consistency errors
- **Sample visualizations**: Magnetogram components and synthesized Stokes profiles
- **Saved data**: Magnetograms, Stokes profiles, ME parameters for each sample

## Fine-tuning Surya for Magnetogram Generation

To fine-tune Surya specifically for magnetogram generation, you can:

1. Use the existing fine-tuning templates (`2_finetune_template_1D.ipynb`)
2. Modify the loss function to include RT consistency terms
3. Use the ME PINN module as a physics-informed loss component

## Notes

- The original `legacy_training.py` uses different line parameters (15648.5 Å, g_eff=3.0) for a different spectral line
- This implementation is specifically adapted for HMI vector magnetograms at 6173.15 Å
- The model can be used for both validation and synthesis of Stokes profiles

## References

- Surya Foundation Model: [GitHub](https://github.com/NASA-IMPACT/Surya)
- Milne-Eddington approximation for solar magnetic field inversions
- HMI Fe I 6173.15 Å line parameters

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or number of samples
2. **Missing scalers**: Run `download_scalers.sh` first
3. **Import errors**: Ensure paths are set correctly (sys.path.append)

### Getting Help

Check the individual README files in subdirectories for more specific documentation.
