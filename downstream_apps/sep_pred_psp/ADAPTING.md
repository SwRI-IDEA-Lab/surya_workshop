# Adapting the Template for Your Own Task

This guide walks you through copying the template fine-tuning app and wiring it to a new
downstream task. Follow the steps in order — each one builds on the previous.

The template task is **solar flare intensity regression** (predicting peak GOES X-ray flux
from SDO image stacks). The numbered scripts and notebooks in this folder give you working
examples of every step.

**The one rule:** everything generic lives in `workshop_infrastructure/` and is *imported*,
never copied. Your app owns only what is specific to your science. If you find yourself
copying a file out of `workshop_infrastructure/`, stop — that is the thing this layout
exists to prevent.

---

## Overview: what the template gives you

```
downstream_apps/template/
├── configs/config_script.yaml       ← single source of truth for all parameters
├── configs.py                       ← FlareDataConfig: the ~4 task-specific config fields
├── 0_dataset_dataloader_template.ipynb
├── 1_baseline_template.ipynb
├── 2_finetune_template_1D.ipynb
├── 3_finetune_template_1D.py        ← runnable training script (derived from notebook 2)
├── ADAPTING.md                      ← this file
├── datasets/
│   └── template_dataset.py          ← FlareDSDataset (extends HelioNetCDFDataset)
├── lightning_modules/
│   └── pl_simple_baseline.py        ← FlareLightningModule (Lightning wrapper)
├── metrics/
│   └── template_metrics.py          ← FlareMetrics (loss + evaluation metrics)
└── models/
    └── simple_baseline.py           ← RegressionFlareModel (linear baseline)
```

And what it *imports* rather than owning:

| From `workshop_infrastructure/` | What it gives you |
|---|---|
| `configs.py` | `DataConfig`, `TrainingConfig`, `ModelConfig`, `load_config()` — the whole config layer |
| `datasets/helio.py` | `HelioNetCDFDataset` — NetCDF loading, local + S3, normalization, frame sampling |
| `datasets/builders.py` | `build_helio_dataloaders()` — maps your config onto ~20 dataset arguments |
| `models/finetune_models.py` | `HelioSpectformer1D` / `HelioSpectformer2D` — backbone plus a configurable head |
| `utils.py` | `build_scalers()`, `apply_peft_lora()`, `load_pretrained_weights()`, S3 checkpoint upload |

---

## Step 1 — Copy the template folder

```bash
cp -r downstream_apps/template downstream_apps/your_task
```

Then rename the classes that have "Flare" or "Template" in their names (Steps 2–5 will
tell you exactly which ones to change).

---

## Step 2 — Declare your task's config fields (`configs.py`)

**File to edit:** `configs.py`.

This file is short by design. It subclasses the shared `DataConfig` with the handful of
fields your task needs, and binds `load_config` to that subclass:

```python
@dataclass
class YourDataConfig(DataConfig):
    your_catalog_path: str = ""
    your_label_column: str = "flux"

    # Any field holding a filesystem path must join this list, so relative values in the
    # YAML resolve against the config file's directory instead of the working directory.
    PATH_FIELDS: ClassVar[tuple[str, ...]] = DataConfig.PATH_FIELDS + ("your_catalog_path",)


load_your_config = partial(load_config, data_cls=YourDataConfig)
```

You inherit every generic field — paths, channels, temporal sampling, all the S3 settings,
`max_samples` — without maintaining a copy of them.

Two behaviors worth knowing:

- **Unknown keys are an error.** If the YAML has a key with no matching field, `load_config()`
  raises and lists the valid names. So add the field here *first*, then the YAML key.
- **Paths resolve relative to the config file**, and `~`/`$VARS` are expanded. Never commit an
  absolute path into someone else's home directory.

---

## Step 3 — Define your dataset (`datasets/`)

**File to edit:** `datasets/template_dataset.py` → rename to `your_task_dataset.py`.

`FlareDSDataset` extends `HelioNetCDFDataset` (in `workshop_infrastructure/datasets/helio.py`).
The base class handles:
- Loading NetCDF files from local disk or S3
- Signum-log normalization per channel
- Input frame sampling and validity filtering

You only need to add your task-specific logic in the subclass:

| What to override | Why |
|---|---|
| `__init__` | Accept your catalog/label source; pass everything else up via `super().__init__(**kwargs)` |
| `__getitem__` | Call `super().__getitem__()` to get the image stack, then attach your label |

> **Normalization: know which space you are in.** The dataset applies signum-log
> compression *and then* a per-channel z-score, so "undoing the transform" is ambiguous.
> `scaler.inverse_transform()` undoes only the z-score, landing in **signum-log** space —
> that is what `destandardize_channels()` gives the linear baseline, and it is usually what
> you want as model input, since raw values span many orders of magnitude.
> `dataset.inverse_transform_data()` undoes **both** stages and returns true **physical**
> units (DN, Gauss), which is what you want for plotting or a physical-space loss.
> The full picture is in the "THE THREE SPACES" block in
> `workshop_infrastructure/datasets/helio.py`.

If your task supplies its own labels (rather than predicting future SDO frames), set
`kwargs.setdefault("load_forecast_frames", False)` before calling `super().__init__()`, as
`FlareDSDataset` does. That stops the loader from fetching future frames it will never use —
worth roughly 1 GB of S3 traffic per sample.

**Key YAML keys that feed into the dataset** (all under `data:`):
```yaml
data:
  train_data_path: ...           # CSV index of NetCDF files (timestep, path, present)
  valid_data_path: ...
  channels: [...]                # Which SDO channels to load
  time_delta_input_minutes: [0]  # Temporal offsets for input frames
  time_delta_target_minutes: 60  # Step size between forecast frames
  s3_anon: true                  # true = public bucket; false = IAM credentials
  s3_mode: download              # download | simplecache | stream
  s3_cache_dir: ~/surya_ws_cache # Required unless s3_mode is "stream"
  max_samples: null              # Cap for quick experiments
```

### Reading from S3

The pre-built indices in `data/indices/` point at `s3://nasa-surya-bench/...`, so `s3_mode`
and `s3_cache_dir` decide how your data actually arrives:

| `s3_mode` | Behavior | Needs `s3_cache_dir`? |
|---|---|---|
| `download` (default) | Fetches each whole file into the cache with parallel multipart, then opens it locally. **Recommended** — NetCDF/HDF5 needs random seeks that streaming cannot serve. | Yes |
| `simplecache` | fsspec read-through cache. | Yes |
| `stream` | Reads directly from S3, nothing written to disk. Works, but measured ~9x slower than `download` on a full SDO frame. Use only when disk space is the binding constraint. | No |

There is deliberately no default cache directory: each SDO file is ~1 GB, so the right
location depends on your machine. Budget ~1 GB × the number of unique timesteps. If your
index has `s3://` paths and no cache directory is set, dataset construction fails
immediately with a suggested path — not twenty minutes into training.

`--s3-cache-dir` overrides it per machine without editing the committed config.

---

## Step 4 — Define your metrics (`metrics/`)

**File to edit:** `metrics/template_metrics.py` → rename to `your_task_metrics.py`.

`FlareMetrics` defines four metric sets selected by the `mode` argument at construction:

| Mode | Purpose | Backpropagates? |
|---|---|---|
| `"train_loss"` | Loss that drives weight updates; logged as `train_loss` | Yes |
| `"val_loss"` | **Logged as `val_loss` — the quantity `ModelCheckpoint` monitors** | No |
| `"train_metrics"` | Extra metrics logged during training | No |
| `"val_metrics"` | Metrics logged at validation, for reporting only | No |

Each method returns `(dict[str, Tensor], list[float])`: a dict of named metric tensors
and a list of weights for combining multiple loss terms.

The dict keys become the metric names in WandB and CSV logs.

> **Which metric selects checkpoints.** `val_loss` does — not `val_metrics`. The names
> invite the opposite guess, so it is worth stating plainly: `val_metrics` is logged for
> reporting and has no effect on which checkpoint is kept.
>
> `FlareMetrics.val_loss` delegates to `train_loss` by default, so out of the box the
> monitored quantity has the same form as the training objective and the two cannot drift
> apart by accident. **Override `val_loss` in your metrics class** when your task needs a
> different validation objective — that is the intended hook, and you should not need to
> touch `lightning_modules/`.
>
> The key is optional: `FlareLightningModule` falls back to `train_loss` if a metrics dict
> has no `"val_loss"` entry.

---

## Step 5 — Define your model head (`models/`)

For 1D output tasks (regression, classification): use `HelioSpectformer1D` from
`workshop_infrastructure/models/finetune_models.py`. It wraps the Surya backbone with a
configurable pooling head and a linear output layer.

For 2D output tasks (pixel-level prediction): use `HelioSpectformer2D`.

The head is fully configured from the YAML `model:` section — you usually don't need to
touch the model code at all, just adjust the config:

```yaml
model:
  pooling: class_token        # class_token | global_average | global_max | attention | transformer
  penultimate_linear_layer: true
  dropout: 0.2
  freeze_backbone: false
  use_lora: true
  lora_config:
    r: 8
    lora_alpha: 8
    ...
```

`use_lora` and `freeze_backbone` together select the fine-tuning regime:

| `use_lora` | `freeze_backbone` | Regime |
|---|---|---|
| `true` | — | LoRA adapters on attention + FFN (default; small trainable count) |
| `false` | `true` | Linear probe — only the head trains |
| `false` | `false` | Full fine-tuning of all 366M parameters |

The training script prints the trainable/total parameter count so you can confirm which
regime you actually got.

If your task needs a custom head (e.g. multi-head output, auxiliary losses), create a new
class in `models/` following the `RegressionFlareModel` pattern in `simple_baseline.py`.

---

## Step 6 — Wire it together in the training script

**File to edit:** `3_finetune_template_1D.py`.

Only two of its four functions have task-specific content:

| Function | What it does | What to change |
|---|---|---|
| `build_datasets` | Calls `build_helio_dataloaders()` | Swap `FlareDSDataset` for your subclass and replace the task-specific kwargs below it |
| `build_model` | Builds `HelioSpectformer1D`, loads weights, applies LoRA / freezing | Usually nothing — driven by `cfg.model` |
| `build_trainer` | Loggers, checkpointing, Lightning Trainer | Nothing |
| `main` | Calls the above in order | Nothing |

`build_datasets` should stay this short — everything generic is already handled:

```python
def build_datasets(cfg):
    return build_helio_dataloaders(
        cfg,
        YourDataset,
        # ↓ only your task's kwargs belong here
        your_catalog_path=cfg.data.your_catalog_path,
        label_transform=_your_label_transform,
    )
```

Also change the `load_flare_config` import to your own `load_your_config`.

---

## Step 7 — Edit `config_script.yaml`

This is the only file you need to edit between experiments. The sections map directly
to the dataclasses:

| YAML section | Python dataclass | Accessed via |
|---|---|---|
| `data:` | your `DataConfig` subclass | `cfg.data.*` |
| `model:` | `ModelConfig` | `cfg.model.*` |
| `model.pretrained_path` | `ModelConfig.pretrained_path` | `cfg.model.pretrained_path` |
| `model.lora_config:` | `LoraAdapterConfig` | `cfg.model.lora_config.*` |
| `model.time_embedding:` | `TimeEmbeddingConfig` | `cfg.model.time_embedding.*` |
| `training:` | flat fields on `TrainingConfig` | `cfg.learning_rate`, `cfg.batch_size`, … |
| `output:` | `OutputConfig` | `cfg.output.*` |
| `logging:` | flat fields on `TrainingConfig` | `cfg.wandb_project`, `cfg.wandb_entity` |

---

## Reproducibility

Two runs of the same config produce the same numbers. That is not free — it is bought by two
keys in the `training:` section, and it is worth understanding what they do before you change
them.

```yaml
training:
  seed: 42             # seeds Python, NumPy, torch, every DataLoader worker, and the shuffle
  deterministic: false # false | warn | true
```

| `deterministic` | Behavior |
|---|---|
| `false` (default) | No guarantees. Two identical runs may disagree. Chosen as the default for throughput. |
| `warn` | Bit-identical wherever a deterministic CUDA kernel exists — which is the whole model as shipped — and a warning naming the op where one does not. Never blocks a run. |
| `true` | Hard guarantee: an op with no deterministic kernel raises. **Incompatible with `model.learned_flow: true`**, which uses `F.grid_sample` (no deterministic CUDA backward). |

Determinism costs roughly **+20% of wall time**: on this template (1 epoch, `max_samples: 10`,
one A100, LoRA) a run takes 159 s with `false` and 190 s with `warn`. That total includes a fixed
~1.8 GB checkpoint load, so the overhead on the compute alone is proportionally larger — budget
more than 20% on long runs. The default trades reproducibility for that throughput.

**Switch to `warn` whenever you are comparing results.** With `false`, you cannot tell whether a
change in your numbers came from your edit or from drift — this config produced `val_loss`
0.060782 and 0.051657 on two runs that differed in nothing at all. Any before/after comparison,
ablation, or hyperparameter sweep should use `warn`.

For a one-off comparison, pass the flag instead of editing the committed config:

```bash
python -m downstream_apps.your_task.3_finetune_template_1D --deterministic warn
```

Note that `seed` still does its job with determinism off: the data order stays pinned, so a
different seed still means a genuinely different run.

Two implementation details you inherit for free, but should not undo:

- **`CUBLAS_WORKSPACE_CONFIG` is set before `import torch`** — at the top of
  `3_finetune_template_1D.py` and in the notebooks' first cell. cuBLAS reads it once at
  initialization, so setting it later silently does nothing. Without it you get a cuBLAS warning
  on every run.
- **The train DataLoader gets an explicit `generator` and `worker_init_fn`** (in
  `workshop_infrastructure/datasets/builders.py`). A bare `shuffle=True` seeds its sampler from
  whatever the global torch RNG state happens to be when the iterator is created — so any code
  you add that consumes RNG beforehand would silently reshuffle your epochs. The worker seeder
  also seeds Python's `random`, because the channel masker in `helio.py` draws from it.

> **Reproducibility is not significance.** With `max_samples: 10` and one epoch the result is
> still noise. This makes it the *same* noise every time, so that a change in the number can be
> attributed to your edit rather than to chance. Before drawing any scientific conclusion, raise
> `max_samples` and vary `seed`.

---

## Quick reference: running the script

```bash
# Full run (--config defaults to this app's configs/config_script.yaml)
CUDA_VISIBLE_DEVICES=0,1 python -m downstream_apps.your_task.3_finetune_template_1D

# Quick sanity check
CUDA_VISIBLE_DEVICES=0 python -m downstream_apps.your_task.3_finetune_template_1D \
    --max-epochs 2 --no-wandb

# Same config, different machine's scratch space
CUDA_VISIBLE_DEVICES=0 python -m downstream_apps.your_task.3_finetune_template_1D \
    --s3-cache-dir /scratch/$USER/helio_cache

# Reproducible run, for comparing a change against a baseline (~20% slower)
CUDA_VISIBLE_DEVICES=0 python -m downstream_apps.your_task.3_finetune_template_1D \
    --deterministic warn
```

Set `max_samples: 10` in the YAML while developing — it limits the dataset so data loading
is fast without changing anything else.

---

## Checklist: files you should have edited

If you touched anything outside this list, ask whether it belongs in
`workshop_infrastructure/` instead:

- [ ] `configs.py` — your `DataConfig` subclass (~15 lines)
- [ ] `configs/config_script.yaml` — your paths and hyperparameters
- [ ] `datasets/your_task_dataset.py` — your labels
- [ ] `metrics/your_task_metrics.py` — your loss and evaluation metrics
- [ ] `3_finetune_template_1D.py` — the two imports and the kwargs in `build_datasets`
- [ ] `models/` — only if you need a custom head
