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

Run the test suite with `pytest tests/ -v` (currently `tests/test_lora_setup.py`, which is CPU-only and fast). For changes not covered by tests, verify by running the training script with `max_samples` capped (see above).

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

PEFT LoRA is applied (rank=8, alpha=8, dropout=0.1) by `apply_peft_lora()` in `workshop_infrastructure/utils.py`.

**Adapted:** `fc1`/`fc2` in all 10 blocks, plus `attn.qkv` and `attn.proj` in the 8 attention blocks — `target_modules: [fc1, fc2, attn.qkv, attn.proj]`. The dotted forms are required: a bare `proj` would also match the Conv2d patch-embedding tokenizer at `embedding.patch_embed.proj`. **Never adapted:** the spectral blocks' `complex_weight`, `attn.to_dynamic_projection`, and the patch embedding.

Surya fuses q/k/v into one `nn.Linear(1280, 3840)`, so one adapter covers all three: ΔW = B·A with B 3840×8 and A 8×1280. q, k and v **share A** and each owns a 1280×8 slice of B, for a combined rank of at most 8 — *not* three independent rank-8 adapters.

PEFT only raises when *no* `target_modules` entry matches anything, so a misspelt name is silently ignored. Verify with the `[LoRA] Adapted modules` list the helper prints at startup.

**The `head_` naming convention.** Every trainable component of a fine-tuning head must be a direct child of the top-level model whose attribute name starts with `head_` (`head_linear`, `head_unembed`, `head_cls_token`, …); the backbone stays at `backbone`. `apply_peft_lora()` discovers those modules and passes them to PEFT as `modules_to_save` so they stay trainable — without this the head is frozen at its random initialization and the adapters fit a random readout. There is no YAML override; the convention *is* the interface. `discover_head_modules()` enforces it at startup and raises an actionable error naming the attribute to rename.

Two constraints follow from how PEFT works, both covered by that validation:
- `modules_to_save` matches module **names**, so a bare `nn.Parameter` on the top-level model cannot be kept trainable. Wrap it in a module — see `ClassToken` in `finetune_models.py` — and read it by **calling** the module, never via attribute access, which under the PEFT wrapper can return the frozen original.
- PEFT matches `modules_to_save` entries with a bare `key.endswith(name)` and **no dot boundary**, so a head name that is a suffix of any backbone module name would wrap that backbone module too.

Three regimes, selected from the `model:` config section:
- `use_lora: true` — LoRA adapters **plus all `head_*` modules** (default). `freeze_backbone` is a no-op here: PEFT freezes everything and then re-enables only adapters and head.
- `use_lora: false, freeze_backbone: true` — linear probe, head only
- `use_lora: false, freeze_backbone: false` — full fine-tuning

Trainable counts for the template config: LoRA 3,157,761 (1,515,520 adapters + 1,642,241 head); probe 1,642,241; full 366M. `tests/test_lora_setup.py` pins all of this.

`HelioSpectformer1D` derives the backbone's `nglo` argument from `pooling` internally (`1` for `class_token`, `0` otherwise) — it is not a config field. `ModelConfig`/`TrainingConfig` also validate several other cross-field invariants at config-load time (`img_size` vs `patch_size`, `spectral_blocks`/`checkpoint_layers` vs `depth`, `time_embedding.time_dim` vs `data.time_delta_input_minutes`, `training.deterministic` vs `model.learned_flow`, and `model.learned_flow` vs `time_embedding.type`) — see `ModelConfig.__post_init__` and `TrainingConfig.__post_init__` in `workshop_infrastructure/configs.py` for the current list, and add new ones there rather than leaving them as documentation-only footguns.

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

**Reproducibility.** `training.seed` and `training.deterministic` (`false` | `warn` | `true`, **default `false`** for throughput — determinism costs ~20% wall time) control it. Results are therefore NOT reproducible out of the box; `warn` is the setting to use when comparing runs. `3_finetune_template_1D.py` sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` **before importing torch** — this is required for deterministic cuBLAS and is inert if moved after the import, so do not "tidy" it into the other imports. The notebooks' first cell does the same. `build_helio_dataloaders()` passes an explicit `generator` and `worker_init_fn`; without them the shuffle order depends on ambient global RNG state. `deterministic: true` is incompatible with `model.learned_flow: true` (`F.grid_sample` has no deterministic CUDA backward); `TrainingConfig.__post_init__` rejects that combination at config-load time.

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
| LoRA application + `head_` discovery | `workshop_infrastructure/utils.py` |
| LoRA setup tests | `tests/test_lora_setup.py` |
| Downstream adapter model | `workshop_infrastructure/models/finetune_models.py` |
| Fine-tuning entry point | `downstream_apps/template/3_finetune_template_1D.py` |
| Model weights (HuggingFace) | `nasa-impact/surya` |
| Pretrained checkpoint | `downstream_apps/template/assets/surya.366m.v1.pt` |
