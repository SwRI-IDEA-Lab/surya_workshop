# Adapting the Template for Your Own Task

This guide walks you through copying the template fine-tuning app and wiring it to a new
downstream task. Follow the steps in order — each one builds on the previous.

The template task is **solar flare intensity regression** (predicting peak GOES X-ray flux
from SDO image stacks). The numbered scripts and notebooks in this folder give you working
examples of every step.

---

## Overview: what the template gives you

```
downstream_apps/template/
├── configs/config_script.yaml       ← single source of truth for all parameters
├── 0_dataset_dataloader_template.ipynb
├── 1_baseline_template.ipynb
├── 2_finetune_template_1D.ipynb
├── 3_finetune_template_1D.py        ← runnable training script (derived from notebook 2)
├── ADAPTING.md                      ← this file
├── configs.py                       ← typed config dataclasses + load_config()
├── datasets/
│   └── template_dataset.py          ← FlareDSDataset (extend HelioNetCDFDataset)
├── lightning_modules/
│   └── pl_simple_baseline.py        ← FlareLightningModule (Lightning wrapper)
├── metrics/
│   └── template_metrics.py          ← FlareMetrics (loss + evaluation metrics)
└── models/
    └── simple_baseline.py           ← RegressionFlareModel (linear baseline)
```

Everything flows through `config_script.yaml`. The training script calls `load_config()`
which returns a fully typed `TrainingConfig` — you never pass raw dicts around.

---

## Step 1 — Copy the template folder

```bash
cp -r downstream_apps/template downstream_apps/your_task
```

Then rename the classes that have "Flare" or "Template" in their names (Steps 2–5 will
tell you exactly which ones to change).

---

## Step 2 — Define your dataset (`datasets/`)

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

**Key YAML keys that feed into the dataset** (all under `data:`):
```yaml
data:
  train_data_path: ...           # CSV index of NetCDF files (timestep, path, present)
  valid_data_path: ...
  channels: [...]                # Which SDO channels to load
  time_delta_input_minutes: [0]  # Temporal offsets for input frames
  time_delta_target_minutes: 60  # Step size between forecast frames
  s3_anon: true                  # true = public bucket; false = IAM credentials
  s3_cache_dir: /path/to/cache   # Required when index contains s3:// paths
  max_samples: null              # Cap for quick experiments
```

If your task has a separate label catalog, add its path and alignment parameters here too
(see `flare_index_path`, `ds_time_column`, `ds_time_tolerance`, `ds_match_direction` in
the template as an example).

---

## Step 3 — Define your metrics (`metrics/`)

**File to edit:** `metrics/template_metrics.py` → rename to `your_task_metrics.py`.

`FlareMetrics` defines three metric sets selected by the `mode` argument at construction:

| Mode | Purpose | Backpropagates? |
|---|---|---|
| `"train_loss"` | Loss that drives weight updates | Yes |
| `"train_metrics"` | Extra metrics logged during training | No |
| `"val_metrics"` | Metrics logged at validation | No |

Each method returns `(dict[str, Tensor], list[float])`: a dict of named metric tensors
and a list of weights for combining multiple loss terms.

The dict keys become the metric names in WandB and CSV logs. `val_loss` (the checkpoint
monitor) is the weighted sum of `val_metrics` outputs — so at least one val metric should
measure prediction quality.

---

## Step 4 — Define your model head (`models/`)

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

If your task needs a custom head (e.g. multi-head output, auxiliary losses), create a new
class in `models/` following the `RegressionFlareModel` pattern in `simple_baseline.py`.

---

## Step 5 — Wire it together in the training script

**File to edit:** `3_finetune_template_1D.py`.

Four functions do all the work:

| Function | What it does | What to change |
|---|---|---|
| `build_datasets` | Instantiates your dataset and wraps it in DataLoaders | Swap `FlareDSDataset` for your subclass; forward any new config fields |
| `build_model` | Builds `HelioSpectformer1D`, loads pretrained weights, applies LoRA | Usually nothing — driven by `cfg.model` |
| `build_trainer` | Sets up loggers, checkpointing, and the Lightning Trainer | Usually nothing |
| `main` | Calls the above in order | Usually nothing |

The script receives a `TrainingConfig` from `load_config()` — use `cfg.*` attributes
everywhere instead of raw dicts.

---

## Step 6 — Update `configs.py`

**File to edit:** `configs.py` (the one in your app folder, not `workshop_infrastructure/configs.py`).

Add any new task-specific fields to `DataConfig` (catalog paths, label parameters, etc.).
`_from_dict()` ensures that adding a field only requires editing the dataclass and YAML —
`load_config()` picks it up automatically.

```python
@dataclass
class DataConfig:
    ...
    your_catalog_path: str        # ← add here
    your_label_column: str = "flux"
```

Then add the corresponding key to `config_script.yaml` under `data:`.

---

## Step 7 — Edit `config_script.yaml`

This is the only file you need to edit between experiments. The five sections map directly
to the dataclasses:

| YAML section | Python dataclass | Accessed via |
|---|---|---|
| `data:` | `DataConfig` | `cfg.data.*` |
| `model:` | `ModelConfig` | `cfg.model.*` |
| `model.pretrained_path` | `ModelConfig.pretrained_path` | `cfg.model.pretrained_path` |
| `model.lora_config:` | `LoraAdapterConfig` | `cfg.model.lora_config.*` |
| `model.time_embedding:` | `TimeEmbeddingConfig` | `cfg.model.time_embedding.*` |
| `training:` | flat fields on `TrainingConfig` | `cfg.learning_rate`, `cfg.batch_size`, … |
| `output:` | `OutputConfig` | `cfg.output.*` |
| `logging:` | flat fields on `TrainingConfig` | `cfg.wandb_project`, `cfg.wandb_entity` |

---

## Quick reference: running the script

```bash
# Full run
CUDA_VISIBLE_DEVICES=0,1 python -m downstream_apps.your_task.3_finetune_template_1D \
    --config downstream_apps/your_task/configs/config_script.yaml

# Quick sanity check (override max_epochs without editing the YAML)
CUDA_VISIBLE_DEVICES=0 python -m downstream_apps.your_task.3_finetune_template_1D \
    --config downstream_apps/your_task/configs/config_script.yaml \
    --max-epochs 2 --no-wandb
```

Set `max_samples: 10` in the YAML while developing — it limits the dataset to 10 samples
so the data loading is fast without changing anything else.
