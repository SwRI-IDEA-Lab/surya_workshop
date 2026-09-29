# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is **surya_workshop**, a standalone repo built around the [Surya](https://github.com/NASA-IMPACT/Surya.git) foundation model for heliophysics (a NASA-IMPACT / IBM AI4Science collaboration). The repo provides:
- `workshop_infrastructure/` — shared config, dataset loaders, dataset/dataloader builders, PEFT utilities, and data pipeline scripts. Also contains a **vendored copy** of the 366M-parameter Surya backbone under `workshop_infrastructure/models/`. There is no `Surya/` submodule: the code was copied in so the repo runs standalone, which means it can drift from upstream without any diff signal.
- `downstream_apps/` — template and concrete downstream fine-tuning applications. `template/` is flare regression (1D output); `Imagetranslation/` is EUV2MAG (2D output, and the worked example of an app that changes the backbone's input channel count)
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
  --batch-size 16 --max-epochs 20

# Quick sanity run (every sample is a ~1 GB download the first time it is seen)
CUDA_VISIBLE_DEVICES=0 python -m downstream_apps.template.3_finetune_template_1D \
  --max-epochs 2 --max-train-samples 8 --no-wandb

# Data-scaling sweep: the validation set is capped separately and stays fixed
for n in 25 50 100 200; do
  CUDA_VISIBLE_DEVICES=0 python -m downstream_apps.template.3_finetune_template_1D \
    --max-train-samples $n --deterministic warn
done

# Benchmark S3 throughput to pick s3_boto3_* settings for this machine
python -m workshop_infrastructure.benchmark_s3 \
  s3://nasa-surya-bench/2011/01/20110131_0000.nc --anon --quick

# Linting / formatting
black --line-length 100 .
isort .
mypy .
```

Run the test suite with `pytest tests/ -v` (CPU-only and fast — about 30 s for the whole suite). For changes not covered by tests, verify by running the training script with `--max-train-samples` capped (see above).

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
- `models/` — **both** models live in the app, not in infrastructure: `simple_baseline.py` (the linear baseline) and `finetune_model.py` (`FlareSuryaModel` = `build_surya_backbone()` + a head written out in app code). Changing the fine-tuning architecture must never require editing `workshop_infrastructure/`; `HelioSpectformer1D`/`2D` remain as reference implementations of all five pooling variants.
- `lightning_modules/` — the training mechanics live in `workshop_infrastructure/lightning_modules/pl_base.py` (`SuryaLightningModule`, and `SuryaFinetuneLightningModule` which overrides `configure_optimizers` only, splitting the learning rate between the randomly-initialized head and the LoRA adapters via `training.head_lr_multiplier` / `training.weight_decay`). An app subclasses them to supply **`target_fn`** — how to get the target tensor out of the batch dict — which is required and has no default, because a wrong target shape broadcasts rather than raising. Flare: `batch["forecast"].unsqueeze(1)` → `(B, 1)`. EUV2MAG: `batch["forecast"].squeeze(2)` → `(B, C, H, W)`
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

Trainable counts for the template config (`pooling: global_average`, `img_size: 1024`): LoRA 3,156,481 (1,515,520 adapters + 1,640,961 head); probe 1,640,961; full ~201M. The head is 1,280 smaller than under `class_token`, which owns a CLS token; the 201M total is the 366M backbone minus the spectral filters that shrink with the token grid. `tests/test_lora_setup.py` pins the module sets.

`nglo_for_pooling()` in `finetune_models.py` derives the backbone's `nglo` argument from `pooling` (`1` for `class_token`, `0` otherwise) — it is not a config field. `build_surya_backbone(cfg.model, **overrides)` is the shared constructor app models call; it fixes `finetune=True` and defaults `nglo` from the pooling.

The default pooling is **`global_average`**, not `class_token`: it adds no parameters and every token carries gradient from the first step, whereas a CLS token must learn what to attend to before it contributes — which a few-hundred-sample fine-tune may never reach. For `global_average` only, `HelioSpectformer1D.forward` and `FlareSuryaModel.forward` pool *before* `head_linear`, which is exactly equivalent (mean and an affine map commute) and applies the layer to one token instead of L. The reorder is **not** valid for any other pooling.

`ModelConfig`/`TrainingConfig` also validate several other cross-field invariants at config-load time (`img_size` vs `patch_size`, `spectral_blocks`/`checkpoint_layers` vs `depth`, `time_embedding.time_dim` vs `data.time_delta_input_minutes`, `training.deterministic` vs `model.learned_flow`, `model.learned_flow` vs `time_embedding.type`, and `model.img_size` vs `data.native_img_size // data.pooling`) — see `ModelConfig.__post_init__` and `TrainingConfig.__post_init__` in `workshop_infrastructure/configs.py` for the current list, and add new ones there rather than leaving them as documentation-only footguns.

### Changing the Backbone's Input Channels

`downstream_apps/Imagetranslation/` feeds Surya 3 of the 13 pretraining channels. Two pieces of infrastructure exist for that case, and both are opt-in:

**The tokenizer is initialized from the pretrained weights, not from scratch.** `adapt_patch_embed_weight()` selects along the channel axis as well as the frame axis, so `(1280, 26, 16, 16)` becomes `(1280, 3, 16, 16)` holding exactly the chosen channels' pretrained planes. The subset **cannot be inferred from the shapes** — `(…, 26, …) → (…, 3, …)` fits several channel/frame splits, each selecting different planes — so `load_pretrained_weights(channel_indices=…, ckpt_in_chans=…)` takes it explicitly and raises without it.

The indices are positions in the **pretraining channel order**, which is neither the app's own `data.channels` nor the key order in `assets/scalers.yaml` (where `hmi_bx/by/bz` precede `hmi_m`). An app records it as a config field (`Euv2MagDataConfig.pretrained_channel_order`) and derives the indices from that; using the wrong list loads the wrong channels' filters with no error.

**The restricted tokenizer must then be trainable**, via `model.trainable_backbone_modules: [embedding.patch_embed]` (empty by default — freezing the backbone is the point of LoRA). `LinearEmbedding` is `patch_embed(x) + pos_embed` with no normalization between them and the blocks are pre-norm, so LayerNorm only ever sees a branch input, never the residual stream. Dropping channels shrinks the tokenizer's output while the fixed-amplitude `pos_embed` does not, and nothing downstream restores that ratio because normalization acts on the sum. A LoRA adapter cannot substitute: it acts *after* tokenization and cannot change a per-channel linear map. Measured for 3-of-13 by `downstream_apps/Imagetranslation/tools/measure_token_scale.py`: token std falls to 0.59 of the 13-channel value, so position's share of each token grows 1.68x.

`resolve_trainable_backbone_modules()` requires each name to match exactly one module, because PEFT matches `modules_to_save` by suffix — the same trap the `head_` convention guards against. Such a layer lands in the **backbone** optimizer group (it starts from pretrained weights, so it wants the base rate, not the head's multiplier).

`DataConfig.validate_with_model()` is the hook for cross-section checks an app needs — EUV2MAG uses it to require `model.in_channels == len(data.input_channels)`.

### Data Pipeline

```
NetCDF files (SDO, 4096×4096, 13 channels, 12-min cadence)
  ↓ CSV index (path, timestamp, label)  ←  data/indices/
  ↓ HelioNetCDFDataset (local, or S3 via data.s3_mode: download | simplecache | stream)
  ↓ Average pooling by data.pooling  →  native_img_size // pooling  (default 4 → 1024²)
  ↓ Signum-log normalization: sign(x)*log(1+|x|) per channel
  ↓ DataLoader → FlareSuryaModel (build_surya_backbone + app head)
```

**Resolution is the main VRAM/speed knob.** `data.pooling` average-pools each frame before
normalization; the token count falls with its square. Measured on one A100 (LoRA,
bf16-mixed, all layers checkpointed): pooling 1 → 4096², 65536 tokens, batch 2, 28.6 GB,
1.72 s/sample; pooling 4 → 1024², 4096 tokens, batch 16, 14.3 GB, 0.099 s/sample — 17×
faster per sample. `model.img_size` must equal `native_img_size // pooling`, validated at
config load. Pooling saves GPU memory and compute, **not** download time: the full
4096×4096 array is read from S3 first and pooled after.

**Pretrained weights are adapted, not dropped.** `load_pretrained_weights()` used to skip
every shape-mismatched key silently and print only a count. Two of them matter, together
~170M of the 366M parameters:
- `embedding.patch_embed.proj.weight` is a Conv2d over `in_chans * time_dim` = 26 channels
  (13 × 2 frames). The template ships `time_dim: 1`, so **the tokenizer was randomly
  initialized on every run**. `adapt_patch_embed_weight()` selects the most-recent-frame
  slice (index `c * T + t`, keeping the tail of the time axis).
- `blocks_spectral_gating.*.filter.complex_weight` is grid-shaped. Because the token grid
  always spans the same field of view, mode *m* is the same physical spatial frequency at
  any resolution, so `truncate_spectral_filter()` takes the low-|k| block in rfft2 layout
  (rows `[0:N/2+1]` + `[-(N/2-1):]`, cols `[0:N/2+1]`). This is a restriction, not an
  interpolation.

Anything else that does not fit now **raises** rather than leaving a backbone tensor at its
random initialization; pass `strict_shapes=False` to opt out deliberately. Both conversions
live in `workshop_infrastructure/models/weight_adaptation.py` and are pinned by
`tests/test_weight_adaptation.py`, including a numerical check that a band-limited signal
is filtered identically at both resolutions.

**Dataset size: `max_train_samples` and `max_val_samples` are separate**, so a data-scaling
study varies the training set while every run is scored on the same validation set.
`HelioNetCDFDataset._apply_subsample()` takes a prefix of a seeded permutation, so subsets
are random *and* nested (the 50-sample subset is contained in the 200-sample one). A
subclass that rebuilds `valid_indices` after `super().__init__()` — as `FlareDSDataset`
does, joining the flare catalog — sets `SUBSAMPLE_IN_BASE_INIT = False` and calls
`_apply_subsample()` itself once its index is final, using the returned position array to
keep parallel frames aligned. `max_samples` remains a shorthand setting both.

Scalers (normalization stats per channel) are stored in `assets/scalers.yaml` and loaded at dataset init time by `build_scalers()`. That function always resolves scaler classes from the vendored `workshop_infrastructure.datasets.transformations`, deliberately ignoring the stale `base:` field each entry records — normalization must not depend on what happens to be installed.

**Three spaces, two different "inverse" operations.** The forward pipeline is signum-log *then* z-score, so:
- `scaler.inverse_transform()` undoes the z-score only → **signum-log** space. This is what `destandardize_channels()` feeds the linear baseline.
- `dataset.inverse_transform_data()` undoes both → **physical** units (DN, Gauss), for plotting or physical-space losses.

Never assume one is the other; the reference block is at the top of `workshop_infrastructure/datasets/helio.py`.

Assets download on first run via `workshop_infrastructure/assets.py:ensure_assets()`. The two `download_*.sh` scripts are thin wrappers over its CLI.

### Configuration

All runtime parameters live in a single YAML file (`configs/config_script.yaml`), parsed by `load_config()` in `workshop_infrastructure/configs.py` into a typed `TrainingConfig`. Sections: `data`, `model` (incl. LoRA and time embedding), `training`, `output`, `logging`.

Three properties matter when editing configs:
- **Adding a key is one edit.** `_TRAINING_KEYS` is derived from `TrainingConfig`'s fields and both sections are splatted into the constructor, so declaring the dataclass field is all it takes. Retired keys get a targeted message: `training.dtype` (which was read and then ignored by the backbone) now errors pointing at `training.precision`.
- **Unknown keys raise.** A key not present on the target dataclass is an error naming the valid alternatives, never a silent no-op. Task-specific keys require a field on the app's `DataConfig` subclass.
- **Paths are relative to the config file** and resolved at load time, so a checked-in config works from any working directory. `s3_cache_dir` is the exception — it expands `~`/`$VARS` but is never anchored to the repo.

**Reproducibility.** `training.seed` and `training.deterministic` (`false` | `warn` | `true`, **default `false`** for throughput — determinism costs ~20% wall time) control it. Results are therefore NOT reproducible out of the box; `warn` is the setting to use when comparing runs. `3_finetune_template_1D.py` sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` **before importing torch** — this is required for deterministic cuBLAS and is inert if moved after the import, so do not "tidy" it into the other imports. The notebooks' first cell does the same. `build_helio_dataloaders()` passes an explicit `generator` and `worker_init_fn`; without them the shuffle order depends on ambient global RNG state. `deterministic: true` is incompatible with `model.learned_flow: true` (`F.grid_sample` has no deterministic CUDA backward); `TrainingConfig.__post_init__` rejects that combination at config-load time.

CLI overrides are deliberately limited to what varies between runs of one config: `--max-epochs`, `--batch-size`, `--max-train-samples` (the data-scaling sweep knob; the validation cap stays in the YAML so it cannot drift between runs), `--s3-cache-dir`, `--deterministic {false,warn,true}`, plus the `--no-wandb` and `--train_baseline` toggles. All of them are applied to `cfg` before anything reads it, so `cfg` is the single record of what ran.

`training.precision` (`bf16-mixed` | `16-mixed` | `32-true`) and `training.accumulate_grad_batches` are config fields rather than hardcoded Trainer arguments. The script still forces `32-true` on CPU.

### Distributed Training

DDP via PyTorch Lightning. Use `CUDA_VISIBLE_DEVICES` to select GPUs. Logging is rank-aware to avoid duplicate WandB/CSV entries.

## Key File Locations

| Purpose | Path |
|---|---|
| Core model architecture (vendored) | `workshop_infrastructure/models/helio_spectformer.py` |
| Pretrained-weight adaptation (resolution, frames, channels) | `workshop_infrastructure/models/weight_adaptation.py` |
| Base dataset loader | `workshop_infrastructure/datasets/helio.py` |
| Dataset/DataLoader builders | `workshop_infrastructure/datasets/builders.py` |
| Config dataclasses + `load_config()` | `workshop_infrastructure/configs.py` |
| Asset download (scalers, weights) | `workshop_infrastructure/assets.py` |
| LoRA application + `head_` discovery | `workshop_infrastructure/utils.py` |
| LoRA setup tests | `tests/test_lora_setup.py` |
| Weight-adaptation tests | `tests/test_weight_adaptation.py` |
| Defaults / subsampling / app-model tests | `tests/test_template_defaults.py` |
| Backbone builder + reference heads | `workshop_infrastructure/models/finetune_models.py` |
| App's editable fine-tuning model | `downstream_apps/template/models/finetune_model.py` |
| Shared LightningModules | `workshop_infrastructure/lightning_modules/pl_base.py` |
| EUV2MAG app (channel-subset example) | `downstream_apps/Imagetranslation/` |
| EUV2MAG handover notes | `downstream_apps/Imagetranslation/MIGRATION.md` |
| Token-scale diagnostic | `downstream_apps/Imagetranslation/tools/measure_token_scale.py` |
| EUV2MAG config tests | `tests/test_euv2mag_config.py` |
| Fine-tuning entry point | `downstream_apps/template/3_finetune_template_1D.py` |
| Model weights (HuggingFace) | `nasa-impact/surya` |
| Pretrained checkpoint | `downstream_apps/template/assets/surya.366m.v1.pt` |
