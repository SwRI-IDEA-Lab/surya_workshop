# EUV2MAG — predicting a magnetogram from EUV images

Fine-tunes the Surya backbone to map three AIA extreme-ultraviolet channels onto an HMI
line-of-sight magnetogram:

```
aia304, aia193, aia171  (t)   ->   hmi_m  (t + 60 min)
```

The scientific question is how much of the photospheric magnetic field is recoverable from
the coronal and chromospheric emission it drives.

> **The target is 60 minutes ahead, not co-temporal.** That is
> `data.time_delta_target_minutes: 60`, carried over from the original pilot config, so this
> is strictly a forecast rather than a pure translation. If you meant the two to be
> simultaneous — which is what "image translation" usually implies, and which isolates the
> EUV-to-field relationship from an hour of evolution — set it to `0`. Worth deciding
> deliberately, because it changes what a result means. The 1x1-convolution baseline
(`--train_baseline`) answers the trivial version of it — how much is predictable per pixel,
with no spatial context — and whatever the fine-tune gains over that is what Surya's
spatial and spectral structure is buying.

> **New to this app after the first workshop?** Read [MIGRATION.md](MIGRATION.md) first. The
> repository changed substantially between the two workshops, and that file maps what you
> wrote onto where it now lives. [RECONCILIATION_PLAN.md](RECONCILIATION_PLAN.md) records
> why the reconciliation was done the way it was.

## Quick start

```bash
conda activate surya_ws
cd <repo root>

# Assets (scalers + the 1.8 GB checkpoint) download on first run. The index points at S3,
# so a cache directory is required — budget ~1 GB per unique timestep.
export SURYA_WS_CACHE_DIR=/scratch/$USER/helio_cache

CUDA_VISIBLE_DEVICES=0 python -m downstream_apps.Imagetranslation.3_finetune_euv2mag
```

Everything is driven by [`configs/config_script.yaml`](configs/config_script.yaml). The CLI
overrides only what varies between runs of one config:

```bash
--max-epochs 5   --batch-size 4   --max-train-samples 500
--s3-cache-dir /scratch/$USER/cache   --deterministic warn
--train_baseline   --no-wandb   --no-figure
```

On a cluster, [`pilot_100.sbatch`](pilot_100.sbatch) wraps the same command:

```bash
sbatch --export=ALL,SURYA_WS_CACHE_DIR=/scratch/$USER/helio_cache pilot_100.sbatch
```

## What this app changes about Surya

Surya was pretrained on 13 input channels. This app feeds it 3, which makes the
**tokenizer** (`backbone.embedding.patch_embed`) the interesting part of the model.

Two things are done about it, and both are deliberate:

**It is initialized from the pretrained weights, not from scratch.** The checkpoint's
patch-embedding convolution is `(1280, 26, 16, 16)` — 13 channels x 2 frames. The three
channels this app uses are selected out of it by position in the *pretraining* channel
order, together with the trailing frame, giving `(1280, 3, 16, 16)`. There is no randomly
initialized layer anywhere in the input path. `load_pretrained_weights` does this, and
raises rather than silently dropping any tensor it cannot adapt.

**It is trainable, unlike the rest of the backbone.** The sliced tokenizer is *not* the
function the backbone was trained to consume. `LinearEmbedding` is
`patch_embed(x) + pos_embed` with no normalization in between, and the blocks are pre-norm,
so LayerNorm only ever sees a branch input and never the residual stream. Taking 3 channels
instead of 13 shrinks the tokenizer's output while `pos_embed` — a fixed-amplitude Fourier
buffer — does not move. Measured on real data:

```
$ python -m downstream_apps.Imagetranslation.tools.measure_token_scale
token std, 13 channels : 2.02 - 2.27
token std,  3 channels : 1.23 - 1.33     ratio 0.59
position's share of each token grows by 1.68x
```

No downstream LayerNorm restores that ratio, because it acts on the sum, and a LoRA adapter
cannot either — it acts *after* tokenization and cannot change a per-channel linear map.
The tokenizer is the only layer that can rescale its own output, so it is kept trainable via
`model.trainable_backbone_modules: [embedding.patch_embed]`. It costs ~984k parameters,
comparable to the LoRA adapters themselves.

To test that decision, set `trainable_backbone_modules: []` and compare — see MIGRATION.md
for the measured comparison.

## Layout

| Path | What it is |
|---|---|
| `configs/config_script.yaml` | Every parameter. Start here. |
| `configs.py` | `Euv2MagDataConfig` — the channel split and the tokenizer index mapping |
| `datasets/euv2mag_dataset.py` | Splits the Surya stack into inputs and target |
| `models/finetune_model.py` | Backbone + decoder head. **Edit the head here.** |
| `models/simple_baseline.py` | The 1x1-conv baseline |
| `metrics/euv2mag_metrics.py` | MSE + MAE |
| `lightning_modules/pl_euv2mag.py` | Target shape only; the training loop is shared |
| `3_finetune_euv2mag.py` | Training entry point |
| `tools/measure_token_scale.py` | The diagnostic behind the trainable tokenizer |

Everything else is imported from `workshop_infrastructure/`, never copied — see
[`downstream_apps/template/ADAPTING.md`](../template/ADAPTING.md).

## Channel lists: four of them, and they answer different questions

This trips people up, and getting it wrong is silent rather than loud.

| Key | Question it answers |
|---|---|
| `data.channels` | What does the dataset decode from each file? (inputs + targets, nothing else) |
| `data.input_channels` | Which of those does the model see? |
| `data.target_channels` | Which does it predict? |
| `data.pretrained_channel_order` | What order was the *checkpoint's* tokenizer trained in? |

The last one is not about this app's data at all — it is what the tokenizer slice indexes
into. It is also **not** the key order in `assets/scalers.yaml`, where `hmi_bx/by/bz`
precede `hmi_m`. Config load rejects every inconsistency it can detect between these lists,
including a channel listed in `data.channels` but never used.

## Notebooks

The interactive path through the same material, in order:

| Notebook | What it covers |
|---|---|
| `0_euv2mag_dataset_dataloader.ipynb` | The dataset and dataloader, the four channel lists, and what the EUV inputs and the magnetogram target actually look like |
| `1_euv2mag_baseline.ipynb` | The 1x1-conv baseline end to end — the number Surya has to beat |
| `2_euv2mag_finetune.ipynb` | The fine-tune: the tokenizer slice, LoRA, and the comparison against the baseline |

Notebooks 1 and 2 are deliberately almost identical — read them side by side, and the
differences are exactly what fine-tuning a foundation model adds. Notebook 2 and
`3_finetune_euv2mag.py` are the same run expressed two ways: same config, same builders,
same model, and with `--deterministic warn` the same numbers.

Every value comes from `configs/config_script.yaml`, so editing the config changes both the
notebooks and the script. While iterating, set `cfg.data.max_train_samples = 8` in the
notebook so an epoch takes seconds — each sample is a ~1 GB read the first time.

Committed without stored outputs, matching the template.
