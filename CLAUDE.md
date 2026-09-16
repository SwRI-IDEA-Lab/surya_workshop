# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is **surya_workshop**, a standalone repo built around the [Surya](https://github.com/NASA-IMPACT/Surya.git) foundation model for heliophysics (a NASA-IMPACT / IBM AI4Science collaboration). The repo provides:
- `workshop_infrastructure/` — shared config, dataset loaders, dataset/dataloader builders, PEFT utilities, and data pipeline scripts. Also contains a **vendored copy** of the 366M-parameter Surya backbone under `workshop_infrastructure/models/`. There is no `Surya/` submodule: the code was copied in so the repo runs standalone, which means it can drift from upstream without any diff signal.
- `downstream_apps/` — template and concrete downstream fine-tuning applications
- `analysis/` — research scripts (embedding probing/ablation); not part of the workshop template path

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
conda env create -f environment.yml
conda activate surya_ws
```

Python 3.12+ required. Key dependencies: PyTorch, PyTorch Lightning, PEFT (LoRA), WandB, SunPy, xarray, Dask, fsspec.

## Common Commands

```bash
# Fine-tune a downstream model (from repo root).
# --config defaults to the app's own configs/config_script.yaml.
CUDA_VISIBLE_DEVICES=0,1 python -m downstream_apps.template.3_finetune_template_1D \
  --batch-size 2 --max-epochs 20

# Quick sanity run: cap the dataset with max_samples in the YAML, then
CUDA_VISIBLE_DEVICES=0 python -m downstream_apps.template.3_finetune_template_1D \
  --max-epochs 2 --no-wandb

# Benchmark S3 throughput to pick s3_boto3_* settings for this machine
python -m workshop_infrastructure.benchmark_s3 \
  s3://nasa-surya-bench/2011/01/20110131_0000.nc --anon --quick

# Linting / formatting
black --line-length 100 .
isort .
mypy .
```

`pytest` is installed in the environment, but no test suite exists yet — verify changes by running the training script with `max_samples` capped (see above).

## Architecture

### Core Model (`workshop_infrastructure/models/`)

**HelioSpectFormer** is a spatiotemporal transformer with two novel block types:

1. **Spectral Gating** (`spectformer.py`): FFT-based global filtering — transforms patches to frequency domain, applies learnable complex weights, then iFFT back.
2. **Long-Short Attention** (`transformer_ls.py`): Combines local windowed attention (`window_size=2`) with global attention via dynamic projection (`dp_rank=4`). Efficient for 4096×4096 solar images.

Input: 13-channel SDO stacks (8 AIA wavelengths + 5 HMI magnetic components), patch size 16, embed_dim 1280.
Architecture: 2 spectral gating blocks + 8 long-short attention blocks.

### Downstream Fine-tuning Pattern

Each downstream task follows this pattern:
- `configs.py` — a `DataConfig` subclass holding **only** the task-specific config fields. Everything generic (and `load_config()` itself) lives in `workshop_infrastructure/configs.py` and is never copied.
- `datasets/` — task dataset inheriting from `HelioNetCDFDataset` (see `workshop_infrastructure/datasets/helio.py`)
- `models/` — task-specific head
- `lightning_modules/` — PyTorch Lightning wrapper with loss and metrics
- `metrics/` — custom metric implementations. Four modes: `train_loss` (backpropagated), `val_loss` (**what ModelCheckpoint monitors**; defaults to `train_loss`), `train_metrics` and `val_metrics` (reported only — they do *not* select checkpoints)
- `configs/config_script.yaml` — single YAML drives everything
- `N_*.py` / `N_*.ipynb` — numbered scripts/notebooks for step-by-step workflow

Dataset and DataLoader construction is **not** re-implemented per app: `build_helio_dataloaders()` in `workshop_infrastructure/datasets/builders.py` maps the config onto the ~20 `HelioNetCDFDataset` arguments, and the app passes only its task-specific kwargs.

### LoRA Fine-tuning

PEFT LoRA is applied to attention and feed-forward layers (rank=8, alpha=8, dropout=0.1, target modules: q/k/v/out_proj, fc1/fc2). The `workshop_infrastructure/utils.py` `apply_peft_lora()` helper handles this.

Three regimes, both selected from the `model:` config section:
- `use_lora: true` — LoRA adapters (default)
- `use_lora: false, freeze_backbone: true` — linear probe, head only
- `use_lora: false, freeze_backbone: false` — full fine-tuning

### Data Pipeline

```
NetCDF files (SDO, 4096×4096, 13 channels, 12-min cadence)
  ↓ CSV index (path, timestamp, label)  ←  data/indices/
  ↓ HelioNetCDFDataset (local, or S3 via data.s3_mode: download | simplecache | stream)
  ↓ Signum-log normalization: sign(x)*log(1+|x|) per channel
  ↓ DataLoader → HelioSpectformer1D → task head
```

Scalers (normalization stats per channel) are stored in `assets/scalers.yaml` and loaded at dataset init time by `build_scalers()`. That function always resolves scaler classes from the vendored `workshop_infrastructure.datasets.transformations`, deliberately ignoring the stale `base:` field each entry records — normalization must not depend on what happens to be installed.

**Three spaces, two different "inverse" operations.** The forward pipeline is signum-log *then* z-score, so:
- `scaler.inverse_transform()` undoes the z-score only → **signum-log** space. This is what `destandardize_channels()` feeds the linear baseline.
- `dataset.inverse_transform_data()` undoes both → **physical** units (DN, Gauss), for plotting or physical-space losses.

Never assume one is the other; the reference block is at the top of `workshop_infrastructure/datasets/helio.py`.

Assets download on first run via `workshop_infrastructure/assets.py:ensure_assets()`. The two `download_*.sh` scripts are thin wrappers over its CLI.

### Configuration

All runtime parameters live in a single YAML file (`configs/config_script.yaml`), parsed by `load_config()` in `workshop_infrastructure/configs.py` into a typed `TrainingConfig`. Sections: `data`, `model` (incl. LoRA and time embedding), `training`, `output`, `logging`.

Two properties matter when editing configs:
- **Unknown keys raise.** A key not present on the target dataclass is an error naming the valid alternatives, never a silent no-op. Task-specific keys require a field on the app's `DataConfig` subclass.
- **Paths are relative to the config file** and resolved at load time, so a checked-in config works from any working directory. `s3_cache_dir` is the exception — it expands `~`/`$VARS` but is never anchored to the repo.

**Reproducibility.** `training.seed` and `training.deterministic` (`false` | `warn` | `true`, **default `false`** for throughput — determinism costs ~20% wall time) control it. Results are therefore NOT reproducible out of the box; `warn` is the setting to use when comparing runs. `3_finetune_template_1D.py` sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` **before importing torch** — this is required for deterministic cuBLAS and is inert if moved after the import, so do not "tidy" it into the other imports. The notebooks' first cell does the same. `build_helio_dataloaders()` passes an explicit `generator` and `worker_init_fn`; without them the shuffle order depends on ambient global RNG state. `deterministic: true` is incompatible with `model.learned_flow: true` (`F.grid_sample` has no deterministic CUDA backward).

CLI overrides are deliberately limited to what varies between runs of one config: `--max-epochs`, `--batch-size`, `--s3-cache-dir`, `--deterministic {false,warn,true}`, plus the `--no-wandb` and `--train_baseline` toggles.

### Distributed Training

DDP via PyTorch Lightning. Use `CUDA_VISIBLE_DEVICES` to select GPUs. Logging is rank-aware to avoid duplicate WandB/CSV entries.

## Key File Locations

| Purpose | Path |
|---|---|
| Core model architecture (vendored) | `workshop_infrastructure/models/helio_spectformer.py` |
| Base dataset loader | `workshop_infrastructure/datasets/helio.py` |
| Dataset/DataLoader builders | `workshop_infrastructure/datasets/builders.py` |
| Config dataclasses + `load_config()` | `workshop_infrastructure/configs.py` |
| Asset download (scalers, weights) | `workshop_infrastructure/assets.py` |
| LoRA application utility | `workshop_infrastructure/utils.py` |
| Downstream adapter model | `workshop_infrastructure/models/finetune_models.py` |
| Fine-tuning entry point | `downstream_apps/template/3_finetune_template_1D.py` |
| Model weights (HuggingFace) | `nasa-impact/surya` |
| Pretrained checkpoint | `downstream_apps/template/assets/surya.366m.v1.pt` |
