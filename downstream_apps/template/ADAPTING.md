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
│   ├── pl_simple_baseline.py        ← FlareLightningModule (Lightning wrapper)
│   └── pl_finetune.py               ← FlareFinetuneLightningModule (head/adapter LRs)
├── metrics/
│   └── template_metrics.py          ← FlareMetrics (loss + evaluation metrics)
└── models/
    ├── simple_baseline.py           ← RegressionFlareModel (linear baseline)
    └── finetune_model.py            ← FlareSuryaModel (Surya backbone + YOUR head)
```

And what it *imports* rather than owning:

| From `workshop_infrastructure/` | What it gives you |
|---|---|
| `configs.py` | `DataConfig`, `TrainingConfig`, `ModelConfig`, `load_config()` — the whole config layer |
| `datasets/helio.py` | `HelioNetCDFDataset` — NetCDF loading, local + S3, normalization, frame sampling |
| `datasets/builders.py` | `build_helio_dataloaders()` — maps your config onto ~20 dataset arguments |
| `models/finetune_models.py` | `build_surya_backbone()` — the backbone; plus `HelioSpectformer1D`/`2D` reference heads |
| `models/weight_adaptation.py` | Converts pretrained weights when you change resolution or frame count |
| `utils.py` | `build_scalers()`, `apply_peft_lora()`, `discover_head_modules()`, `load_pretrained_weights()`, S3 checkpoint upload |

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

**File to edit:** `models/finetune_model.py`.

`FlareSuryaModel` is the app's own fine-tuning model, and it is the counterpart of
`simple_baseline.py`: the backbone comes from `build_surya_backbone()` in the
infrastructure (nobody should re-type Surya's 18 constructor arguments), and everything
after it — the pooling, the head layers, the forward pass — is in your app for you to
edit. **Changing the architecture never requires touching `workshop_infrastructure/`.**

```python
class FlareSuryaModel(nn.Module):
    def __init__(self, model_cfg, num_outputs=1, **backbone_overrides):
        super().__init__()
        self.backbone = build_surya_backbone(model_cfg, nglo=0, **backbone_overrides)
        # ---- yours ----
        self.head_linear  = nn.Linear(model_cfg.embed_dim, model_cfg.embed_dim)
        self.head_dropout = nn.Dropout(model_cfg.dropout)
        self.head_unembed = nn.Linear(model_cfg.embed_dim, num_outputs)

    def pool(self, tokens):          # (B, L, D) -> (B, D)
        return tokens.mean(dim=1)
```

The default is mean pooling over the patch tokens (`model.pooling: global_average`),
because it adds no parameters and every token gets gradient from the first step — whereas
a class token has to learn what to attend to before it contributes anything, which a
short fine-tune on a few hundred samples may never reach. Swapping the pooling is one
method. `HelioSpectformer1D` in `workshop_infrastructure/models/finetune_models.py` has
working implementations of all five variants (`global_average`, `global_max`,
`class_token`, `attention`, `transformer`) to copy from; note that `class_token` also
needs `nglo=1` and `backbone.forward_with_cls_token()`.

For 2D output tasks (pixel-level prediction), `HelioSpectformer2D` is the equivalent
starting point.

### The optimizer: `lightning_modules/pl_finetune.py`

`FlareFinetuneLightningModule` subclasses the baseline module and overrides only
`configure_optimizers`. Under LoRA the model holds two populations of parameters: the
adapters, which start as a near-identity perturbation of a backbone that already works,
and the head, which starts *random*. One learning rate cannot serve both — low enough for
the adapters and the head crawls, high enough for the head and the adapters shove the
pretrained features around in the first few steps. `training.head_lr_multiplier` (default
10) gives the head a larger rate; `training.weight_decay` applies to the adapters only,
where pulling toward zero means pulling toward the *pretrained* weights.

Set `head_lr_multiplier: 1.0` and `weight_decay: 0.0` to get plain Adam over everything.

### Fine-tuning regimes

`use_lora` and `freeze_backbone` together select the regime:

| `use_lora` | `freeze_backbone` | Regime | Trainable (1024² config) |
|---|---|---|---|
| `true` | ignored | LoRA adapters on FFN + attention, **plus the whole head** (default) | 3,156,481 |
| `false` | `true` | Linear probe — only the head trains | 1,640,961 |
| `false` | `false` | Full fine-tuning | ~201M |

`freeze_backbone` has no effect when `use_lora: true`: PEFT freezes every parameter and
then re-enables the adapters and the head regardless.

Adapters go on `fc1`/`fc2` in all ten blocks and on `attn.qkv`/`attn.proj` in the eight
attention blocks. The spectral `complex_weight`, `attn.to_dynamic_projection`, and the
patch embedding are never adapted. Because Surya fuses q/k/v into a single
`nn.Linear(1280, 3840)`, one adapter covers all three: they share the `8×1280` matrix `A`
and each takes its own `1280×8` slice of `B`, so their combined rank is at most 8 — not
three independent rank-8 adapters.

The training script prints the trainable/total parameter count, the list of adapted
modules, and the trainable head modules, so you can confirm which regime you actually got.

> **⚠️ LoRA results from before this was fixed are invalid.** Earlier runs passed no
> `modules_to_save` to PEFT, so the head stayed frozen at its random initialisation and the
> adapters were fitted to a random readout; the loss still went down. Those runs also
> targeted `q_proj`/`k_proj`/`v_proj`/`out_proj`, which do not exist in this backbone, so
> attention was never adapted. Re-run any LoRA experiment, including regime comparisons.
> Fine-tuned checkpoints from before the fix can no longer be loaded, because the head
> attributes were renamed (`linear` → `head_linear`, and so on). Pretrained Surya weights
> are unaffected — that checkpoint contains only backbone keys.

### Adding your own head layers: the `head_` rule

If your task needs a custom head (e.g. multi-head output, auxiliary losses), create a new
class in `models/` following the `RegressionFlareModel` pattern in `simple_baseline.py`.

**Any trainable head component must be a direct attribute of the top-level model whose
name starts with `head_`** — `self.head_linear`, `self.head_unembed`, `self.head_cls_token`.
The backbone stays at `self.backbone`. `apply_peft_lora()` discovers everything with that
prefix and tells PEFT to keep it trainable; a head layer without the prefix is silently
frozen, which is exactly the bug described above. The rule is enforced at startup, and the
error names the attribute to rename, so you will not discover this from a loss curve.

Two consequences worth knowing:

- **Never use a bare `nn.Parameter` at the top level.** PEFT matches module *names*, so a
  loose parameter cannot be kept trainable. Wrap it in a small module — see `ClassToken` in
  `workshop_infrastructure/models/finetune_models.py` — and read it by **calling** the
  module (`self.head_cls_token(batch_size)`). Attribute access on the PEFT wrapper can hand
  back the frozen original. For the same reason, call head modules with **positional**
  arguments: PEFT's wrapper requires at least one.
- **Do not reuse a name that ends a backbone module's name.** PEFT matches with a bare
  `endswith` and no dot boundary, so a head called `head_linear` would also capture a
  backbone module named `my_head_linear`. This is checked at startup too.

---

## Step 6 — Wire it together in the training script

**File to edit:** `3_finetune_template_1D.py`.

Only two of its four functions have task-specific content:

| Function | What it does | What to change |
|---|---|---|
| `build_datasets` | Calls `build_helio_dataloaders()` | Swap `FlareDSDataset` for your subclass and replace the task-specific kwargs below it |
| `build_model` | Builds `HelioSpectformer1D`, loads weights, applies LoRA / freezing | Usually nothing — driven by `cfg.model`. A custom head must follow the `head_` rule above |
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

Adding a `training:` or `logging:` key is one edit — declare the field on `TrainingConfig`
in `workshop_infrastructure/configs.py`. The accepted key list is derived from the
dataclass, so it cannot fall out of sync. A `data:` key goes on your `DataConfig` subclass.

### Resolution: `data.pooling` and `model.img_size`

`data.pooling` average-pools each frame before normalization, so the model sees
`native_img_size // pooling` pixels. It is the main memory and speed knob, because the
token count falls with its square. Measured on one A100 (LoRA, bf16-mixed):

| `pooling` | frames | tokens | batch | peak VRAM | s/sample |
|---|---|---|---|---|---|
| 1 | 4096² | 65536 | 2 | 28.6 GB | 1.72 |
| 2 | 2048² | 16384 | 8 | 27.9 GB | 0.40 |
| **4** | **1024²** | **4096** | **16** | **14.3 GB** | **0.099** |
| 4 | 1024² | 4096 | 32 | 27.7 GB | 0.099 |

`model.img_size` must equal `native_img_size // pooling`; the config load checks it, so a
mismatch is a named error rather than a reshape failure inside the first forward pass.

Lowering the resolution does **not** throw the pretrained model away.
`load_pretrained_weights()` restricts the two spectral-gating filters to the smaller
token grid instead of dropping them (~170M parameters). The token grid always spans the
same field of view, so mode *m* is the same physical spatial frequency at any resolution
and the conversion is a truncation to the frequencies the smaller grid can represent —
see `workshop_infrastructure/models/weight_adaptation.py`.

Pooling saves GPU memory and compute, **not** download time: the full 4096×4096 array is
read from S3 first and pooled after.

### Dataset size: separate train and validation caps

`data.max_train_samples` and `data.max_val_samples` are independent, so you can vary the
amount of training data and score every run on the same validation set. The subset is a
seeded random sample (not the earliest *N* events) and subsets **nest**: with the same
`training.seed`, the 50-sample run's data is a subset of the 200-sample run's, so the
difference between two points on a scaling curve is *more* data, not *different* data.

`data.max_samples` is a shorthand that sets both when the specific keys are absent.

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

# Data-scaling sweep: the validation set is capped separately and stays fixed
for n in 25 50 100 200; do
  CUDA_VISIBLE_DEVICES=0 python -m downstream_apps.your_task.3_finetune_template_1D \
      --max-train-samples $n --deterministic warn
done
```

Set `max_train_samples: 8` in the YAML while developing — it limits the dataset so data
loading is fast without changing anything else. Every sample is a ~1 GB download the
first time it is seen, so a small cap is the difference between a 30-second iteration
and a 20-minute one.

---

## If your task changes the backbone's input channels

Most forks change the labels and leave the 13-channel input stack alone. If yours feeds
Surya a *subset* of those channels — as `downstream_apps/Imagetranslation/` does, mapping
three EUV channels onto a magnetogram — two things need saying explicitly, and both are
opt-in because they are wrong for every other app.

**1. Tell `load_pretrained_weights` which pretrained channels yours are.**

```python
load_pretrained_weights(
    model, cfg.model.pretrained_path,
    channel_indices=cfg.data.input_channel_indices,      # positions in the PRETRAINING order
    ckpt_in_chans=len(cfg.data.pretrained_channel_order),
)
```

The tokenizer is then built at your channel count and initialized from exactly those
channels' pretrained filters, instead of from scratch. Without `channel_indices` the load
raises rather than guessing: `(1280, 26, p, p) -> (1280, 3, p, p)` is consistent with
several channel/frame splits, each selecting different planes.

> **The indices are positions in the pretraining channel order** — not in your
> `data.channels`, and not in `assets/scalers.yaml` (where `hmi_bx/by/bz` precede `hmi_m`).
> Record the pretraining order as a field on your `DataConfig` subclass and derive the
> indices from it. Getting this wrong loads three *other* channels' filters, with no error
> and nothing in the loss curve to say so.

**2. Keep the tokenizer trainable.**

```yaml
model:
  trainable_backbone_modules: [embedding.patch_embed]
```

The restricted tokenizer is not the function the backbone was trained to consume.
`LinearEmbedding` is `patch_embed(x) + pos_embed` with no normalization in between, and the
blocks are pre-norm, so LayerNorm only ever sees a branch input and never the residual
stream. Fewer channels means a smaller tokenizer output, while `pos_embed` is a
fixed-amplitude buffer that does not shrink with it — so position's share of every token
grows, and normalization cannot undo that because it acts on the sum. Only the tokenizer can
rescale its own output; a LoRA adapter acts *after* tokenization and cannot change a
per-channel linear map.

Measure it for your channel set before deciding:

```bash
python -m downstream_apps.Imagetranslation.tools.measure_token_scale
```

For 3 of 13 it is a factor of 1.68 in position's share, at a cost of ~984k trainable
parameters. Such a layer trains at the base learning rate, not the head's multiplier, since
it starts from pretrained weights.

**3. Validate the two against each other.** Override `validate_with_model()` on your
`DataConfig` so `model.in_channels` and `len(data.input_channels)` cannot disagree — that
mismatch otherwise surfaces as an opaque error in the first forward pass. See
`downstream_apps/Imagetranslation/configs.py`.

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
- [ ] `lightning_modules/` — normally just `target_fn`, i.e. how to get the target out
      of the batch dict; the training loop itself is inherited from
      `workshop_infrastructure/lightning_modules/pl_base.py`
