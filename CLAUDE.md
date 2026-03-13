# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is **surya_workshop**, a monorepo wrapping the [Surya](https://github.com/NASA-IMPACT/Surya.git) foundation model for heliophysics (a NASA-IMPACT / IBM AI4Science collaboration). The repo provides:
- `Surya/` — git submodule containing the core 366M-parameter spatiotemporal transformer model
- `downstream_apps/` — template and concrete downstream fine-tuning applications
- `workshop_infrastructure/` — shared dataset loaders, PEFT utilities, and data pipeline scripts

The objective of this repo is to allow future Surya users an easy to modify set of templates that they can use to build their own finetunign applications.  Most of the reusable infrastructure should be in the `workshop_infrastructure/` folder. 

The primary directives of any code development should be:

1. Clarity.
2. Reusability.
3. Simplicity.
4. Functionality

As a secondary objective, this repository should help people develop good AI development
practices in scientific AI.

## Environment Setup

```bash
# Conda (primary)
conda env create -f environment.yml
conda activate surya_ws

# Or with uv (from Surya/ directory)
uv sync
source .venv/bin/activate
```

Python 3.11+ required. Key dependencies: PyTorch, PyTorch Lightning, PEFT (LoRA), WandB, SunPy, xarray, Dask, fsspec.

## Common Commands

```bash
# Run the end-to-end model validation test (downloads model + data from HuggingFace)
cd Surya
python -m pytest -s -o log_cli=true tests/test_surya.py

# Run a single test
python -m pytest -s -k "test_name" tests/test_surya.py

# Download sample training data for downstream examples
cd Surya/downstream_examples
python download_sample_train_data.py

# Fine-tune a downstream model (from repo root)
CUDA_VISIBLE_DEVICES=0,1 python -m downstream_apps.template.3_finetune_template_1D \
  --config downstream_apps/template/configs/config_script.yaml \
  --batch-size 2 --max-epochs 20

# Linting / formatting (configured in Surya/pyproject.toml)
black --line-length 100 .
isort .
mypy .
```

## Architecture

### Core Model (`Surya/surya/models/`)

**HelioSpectFormer** is a spatiotemporal transformer with two novel block types:

1. **Spectral Gating** (`spectformer.py`): FFT-based global filtering — transforms patches to frequency domain, applies learnable complex weights, then iFFT back.
2. **Long-Short Attention** (`transformer_ls.py`): Combines local windowed attention (`window_size=2`) with global attention via dynamic projection (`dp_rank=4`). Efficient for 4096×4096 solar images.

Input: 13-channel SDO stacks (8 AIA wavelengths + 5 HMI magnetic components), patch size 16, embed_dim 1280.
Architecture: 2 spectral gating blocks + 8 long-short attention blocks.

### Downstream Fine-tuning Pattern

Each downstream task follows this pattern:
- `datasets/` — task dataset inheriting from `HelioNetCDFDatasetAWS` (see `workshop_infrastructure/datasets/helio_aws.py`)
- `models/` — task-specific head
- `lightning_modules/` — PyTorch Lightning wrapper with loss and metrics
- `metrics/` — custom metric implementations
- `configs/config_script.yaml` — single YAML drives everything
- `N_*.py` / `N_*.ipynb` — numbered scripts/notebooks for step-by-step workflow

### LoRA Fine-tuning

PEFT LoRA is applied to attention and feed-forward layers (rank=8, alpha=8, dropout=0.1, target modules: q/k/v/out_proj, fc1/fc2). The `workshop_infrastructure/utils.py` `apply_lora()` helper handles this. Backbone can optionally be frozen via config.

### Data Pipeline

```
NetCDF files (SDO, 4096×4096, 13 channels, 12-min cadence)
  ↓ CSV index (path, timestamp, label)  ←  data/indices/
  ↓ HelioNetCDFDatasetAWS (fsspec simplecache for S3 → local)
  ↓ Signum-log normalization: sign(x)*log(1+|x|) per channel
  ↓ DataLoader → HelioSpectformer1D → task head
```

Scalers (normalization stats per channel) are stored in `assets/scalers.yaml` and loaded at dataset init time.

### Configuration

All runtime parameters live in a single YAML file (`configs/config_script.yaml`). Structured into `DataConfig`, model config, LoRA config, training config, and output config sections. CLI args (`--batch-size`, `--max-epochs`, S3 flags) override YAML values at runtime.

### Distributed Training

DDP via PyTorch Lightning. Use `CUDA_VISIBLE_DEVICES` to select GPUs. FSDP is also available in `surya/utils/distributed.py`. Logging is rank-aware to avoid duplicate WandB/CSV entries.

## Key File Locations

| Purpose | Path |
|---|---|
| Core model architecture | `Surya/surya/models/helio_spectformer.py` |
| Base dataset loader | `workshop_infrastructure/datasets/helio_aws.py` |
| LoRA application utility | `workshop_infrastructure/utils.py` |
| Downstream adapter model | `workshop_infrastructure/models/finetune_models.py` |
| Fine-tuning entry point | `downstream_apps/template/3_finetune_template_1D.py` |
| Model weights (HuggingFace) | `nasa-impact/surya` |
| Pretrained checkpoint | `downstream_apps/template/assets/surya.366m.v1.pt` |
