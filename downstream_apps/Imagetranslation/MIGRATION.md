# Finding your way back into EUV2MAG

You wrote this app at the first Surya workshop. The repository was restructured
substantially between then and the second one, and this file exists so you do not have to
reconstruct what happened from sixty commits of diff.

Your science is intact. The plumbing around it is no longer yours to maintain.

---

## 1. What happened, briefly

- The `Surya/` **submodule is gone**. The backbone was copied into
  `workshop_infrastructure/models/`, so the repo runs standalone.
- **Config is typed and fails loudly.** There is no `config["model"]["img_size"]` any more;
  there is `cfg.model.img_size`, built by `load_config()` from one YAML. An unrecognized key
  is an error naming the valid alternatives, not a silent no-op.
- **Dataset, DataLoader, model, checkpoint loading and LoRA construction moved into
  `workshop_infrastructure/`.** Each of these you had written by hand; each is now one call.
- Assets download themselves. S3 access is one `s3_mode` setting. There are tests.

The reasoning behind the specific choices made for *this* app is in
[RECONCILIATION_PLAN.md](RECONCILIATION_PLAN.md).

## 2. Reading order

Roughly an hour, in this order:

1. **`configs/config_script.yaml`** — everything is here now. Read the comments; they carry
   most of what changed.
2. **`configs.py`** — the four channel lists and why they are separate.
3. **`3_finetune_euv2mag.py`** — the entry point. Compare `build_datasets` and `build_model`
   against what your pilot script did by hand.
4. **`datasets/euv2mag_dataset.py`** — your channel split, unchanged in substance.
5. **`models/finetune_model.py`** — the backbone + head, and the long comment about the
   tokenizer. This is the file that most repays reading.
6. **`README.md`** — how to run it.
7. `../template/ADAPTING.md` and `../../CLAUDE.md` when you want the general contract.

**You should not need to read** `workshop_infrastructure/datasets/helio.py` or
`builders.py`. They are long, they are shared with every other app, and the app never calls
into them except through `build_helio_dataloaders`. If you find yourself in there to get
something done, that is a sign the infrastructure is missing a knob — say so rather than
working around it.

## 3. Where your code went

| What you wrote | Where it is now |
|---|---|
| `EUV2MAGDataset` channel split | `datasets/euv2mag_dataset.py`, same logic; the per-item validation moved to config load and the indices are resolved once in `__init__` |
| `Conv2DImageTranslationModel` | `models/simple_baseline.py`. Now takes the batch dict, so it no longer needs `TensorOnlyLightningModule` to unwrap it |
| `ImageTranslationMetrics` | `metrics/euv2mag_metrics.py`, plus the `val_loss` mode it was missing — `ModelCheckpoint` monitors that, and without it checkpoints were selected on the training objective |
| `ImageTranslationLightningModule` | `lightning_modules/pl_euv2mag.py`, three lines over a shared base |
| `ChannelAdapter` (Conv3d 3→13) | Retired — see §5 |
| the 157-tensor checkpoint loop, and `assert len(matched) == 157` | `load_pretrained_weights()` |
| dataset + DataLoader construction | `build_helio_dataloaders()` |
| `max_number_of_samples=100/20`, `sampling_seed=42` | `data.max_train_samples` / `data.max_val_samples` + `training.seed` |
| hardcoded `lr=1e-3`, `max_epochs=1`, `batch_size=1`, `num_workers=0`, wandb entity, `precision` | `configs/config_script.yaml` |
| `3_euv2mag_100train_20val_pilot.py` | `3_finetune_euv2mag.py`. The 100/20 regime is config now, not a filename |
| `setup_conda_data001.sh` | `workshop_infrastructure/setup_scripts/setup_conda_shared_storage.sh`, with `SURYA_WS_BASE` / `SURYA_WS_ENV` instead of `/data001/finetuning_dh` and `surya_dhegde` |
| `pilot_100.sbatch` | Same name, no absolute paths; needs `SURYA_WS_CACHE_DIR` |

Your evaluation figure survived, in `save_prediction_figure()` at the bottom of the training
script — including the inverse transform to Gauss, which is the part that makes it a
physical result rather than a picture of normalized numbers.

## 4. What was deleted, and why it is safe

**`configs/config.yaml`** was an active-region segmentation config, not this app's. Its
`job_id` was `AR-segmentation_lora_r32_a64_d0.1` (note `r32`, while the file set `r: 8`
below it), and it carried `ar_index_train/valid/test`, `select: bce`, `dice:`, `iou:`,
`bce:`, and an `adapter:` block naming `['aia94','aia131','aia171']` — not the channels the
pilot actually used. Nothing read any of those keys. `optimizer.learning_rate`,
`optimizer.max_epochs` and `freeze_backbone` were in it and also never read; the pilot
hardcoded its own values.

**`datasets/template_dataset.py`** (`Euv2MagDSDataset`) was the flare template with the class
renamed and two docstring lines changed. Every line of logic was flare-catalog matching. It
was imported nowhere.

Also removed: the unused `RegressionFlareModel`, `FlareLightningModule` and `FlareMetrics`
copies that came along when the template was forked.

## 5. Three things that were silently wrong

This bears on numbers you may already have shown people.

**LoRA never touched the attention layers.** Your config said:

```yaml
target_modules: ['q_proj', 'v_proj', 'k_proj', 'out_proj', 'fc1', 'fc2']
```

The Surya backbone has no such modules. `AttentionLS` uses a fused `self.qkv` and
`self.proj` (`workshop_infrastructure/models/transformer_ls.py:47-49`). PEFT raises only
when *no* target matches anything, and `fc1`/`fc2` did match — so four of the six names
silently matched nothing, and every "LoRA fine-tune" adapted the feed-forward layers only.
The config now uses `[fc1, fc2, attn.qkv, attn.proj]`, and `test_euv2mag_config.py` pins it.
This was `main`'s default too, not something you got wrong.

**The tokenizer.** Your checkpoint loop reshaped `(1280, 26, 16, 16)` to `(1280, 13, 16, 16)`
by summing over the time axis. That instinct was right, and better than what `main` did —
`main` dropped the tensor silently on a shape mismatch and started from a random tokenizer.
Both are superseded: the tokenizer is now built for 3 channels and initialized by *selecting*
the three channels' pretrained planes at the trailing frame, and `strict_shapes=True` makes
a silent drop impossible.

**`ImageTranslationLightningModule` defined `forward` twice.** Python kept the second, so
the behaviour was fine, but the surviving docstring described the dead one. Also,
`sync_dist=True` had been dropped from the `self.log` calls — harmless on one GPU, but it
under-reports every metric under DDP. Both fixed.

## 6. Why the ChannelAdapter was retired

Your `ChannelAdapter` put a `nn.Conv3d(3, 13, kernel_size=1)` in front of an unmodified
13-channel tokenizer. That preserves every pretrained tokenizer weight exactly — which is
the right instinct, and it is why the input path got explicit attention at all.

The cost is that the Conv3d is randomly initialized, so at step 0 the composite is *not* the
pretrained function either: the pretrained tokenizer is being fed three channels smeared
into thirteen slots by a random matrix. It also adds a layer rather than adapting one.

Selecting the channels carries the same idea through to the weights. The tokenizer becomes
exactly the pretrained tokenizer restricted to `aia304`, `aia193`, `aia171`, with no random
layer anywhere in the input path — and it is then trainable, so it can re-fit what the
restriction changed. Which brings us to:

**The restricted tokenizer has to train, and this is measurable.** `LinearEmbedding` is
`patch_embed(x) + pos_embed` with no normalization in between, and the blocks are pre-norm,
so LayerNorm only ever sees a branch input and never the residual stream. Take 3 channels
instead of 13 and the tokenizer's output shrinks while `pos_embed`, a fixed-amplitude
Fourier buffer, does not:

```
$ python -m downstream_apps.Imagetranslation.tools.measure_token_scale
token std, 13 channels : 2.02 - 2.27
token std,  3 channels : 1.23 - 1.33      ratio 0.59
position's share of each token grows by 1.68x
```

Nothing downstream restores that ratio — normalization acts on the sum — and a LoRA adapter
cannot, because it acts after tokenization and cannot change a per-channel linear map. Only
the tokenizer can rescale its own output. Hence
`model.trainable_backbone_modules: [embedding.patch_embed]`, which costs ~984k parameters,
about what the LoRA adapters themselves cost.

<!-- AB_RESULTS -->

## 7. Things you can do now that you could not before

- **`data.pooling`** — average-pools each frame, so `pooling: 4` trains at 1024x1024 with the
  token count down 16x. This is why `batch_size: 1` is no longer forced on you. No pretrained
  weights are lost: the spectral filters are restricted to the smaller token grid.
- **Independent train/val caps** that nest across sizes, so a data-scaling curve measures
  adding data rather than swapping it, and every run is scored on the same validation set.
- **`training.deterministic: warn`** — bit-identical runs, so a change in your numbers is
  attributable to your edit rather than to run-to-run drift. Verified: two runs of this app
  give the same `val_loss` to six decimals.
- **`data.s3_mode`** — `download` (default, fastest), `simplecache`, or `stream`.
- **Assets download themselves** on first run.
- **`pytest tests/`** — the suite covers the tokenizer slice, the LoRA setup and this app's
  config validation.

## 8. Running it

```bash
conda activate surya_ws
export SURYA_WS_CACHE_DIR=/scratch/$USER/helio_cache

# From the repo root.
CUDA_VISIBLE_DEVICES=0 python -m downstream_apps.Imagetranslation.3_finetune_euv2mag

# A quick check that everything is wired up, before committing real time to it:
CUDA_VISIBLE_DEVICES=0 python -m downstream_apps.Imagetranslation.3_finetune_euv2mag \
  --max-epochs 1 --max-train-samples 4 --no-wandb --deterministic warn

# The baseline, for the comparison that tells you what Surya is buying:
CUDA_VISIBLE_DEVICES=0 python -m downstream_apps.Imagetranslation.3_finetune_euv2mag \
  --train_baseline

# On the cluster:
sbatch --export=ALL,SURYA_WS_CACHE_DIR=/scratch/$USER/helio_cache pilot_100.sbatch
```

To re-run the tokenizer A/B yourself, set `model.trainable_backbone_modules: []` in the YAML
and compare against the default. `[MODEL] Trainable parameters` should differ by 984,320.

## 9. Still to do

**The three notebooks do not run.** `0_euv2mag_dataset_dataloader.ipynb`,
`1_euv2mag_baseline.ipynb` and `2_euv2mag_finetune.ipynb` still import
`EUV2MAG_Dataset`, `helio_boto` and the deleted `config.yaml`. They were deliberately left
for a second pass rather than half-converted, because the script path had to be settled
first. When they are rewritten:

- Notebook 0's per-channel EUV and inverse-transformed HMI plots are worth keeping — they
  are the reason that notebook exists.
- Notebook 1 no longer needs `TensorOnlyLightningModule`: the baseline takes the batch dict
  now, so it shares the Lightning module with the fine-tune.
- Notebook 2 should mirror `3_finetune_euv2mag.py` and import from the app's modules rather
  than retyping them, so the two cannot drift.

**One thing not carried over:** your `environment.yml` pins (`s3fs==2024.6.1`,
`fsspec==2024.6.1`, `datasets==2.21.0`, `boto3==1.41.5`, `pyarrow=23.*`). They fixed a real
resolution failure at the time, but the current file already pins those packages
deliberately to newer versions (`s3fs==2026.1.*`, `boto3==1.42.*`), and `fsspec` and
`pyarrow` now resolve transitively at correct versions. Carrying the old pins forward would
have been a two-year downgrade to fix a problem that no longer exists. If you hit the
resolution failure again, say so — that would mean the current pins are wrong.
