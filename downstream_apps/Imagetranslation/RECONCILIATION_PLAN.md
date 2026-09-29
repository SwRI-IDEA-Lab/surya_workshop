# Reconciling EUV2MAG with the second-workshop infrastructure

**Plan of record, executed 2026-09-28.** This is a historical document, not live
instructions. It records what was done to bring this app forward from the first Surya
workshop and — more usefully — *why* each choice was made, since that reasoning is the part
nobody can reconstruct from the diff later.

For finding your way around the result, read [MIGRATION.md](MIGRATION.md) instead.

---

## The situation

`downstream_apps/Imagetranslation/` was built during the first workshop, forked from the
template as it stood at merge-base `fe1b095`. It maps three EUV channels onto one
magnetogram channel — a subset of the 13 pretraining input channels to a single output.

Between the two workshops the shared infrastructure was rewritten: the `Surya/` submodule
removed and the backbone vendored in, the config layer replaced by a typed `TrainingConfig`
hierarchy with fail-loud validation, dataset and dataloader construction moved into
`build_helio_dataloaders()`, S3 access consolidated behind `s3_mode`, assets made
self-downloading. The app therefore imported modules that no longer existed (`helio_boto`,
`helio_aws`), read a raw YAML dict no loader produced, and hand-rolled dataset, dataloader,
model, checkpoint-loading and LoRA code that infrastructure had come to own.

It also carried real scientific development that had to survive: the channel-subsetting
dataset, the image-translation head and metrics, the Conv1x1 baseline, the 100-train/20-val
pilot, and the evaluation figure.

## Three findings that shaped the work

**1. `main` silently discarded the pretrained tokenizer.** The checkpoint's
`embedding.patch_embed.proj.weight` is `(1280, 26, 16, 16)` — 13 channels x 2 frames. With
`time_dim: 1` the model wants `(1280, 13, 16, 16)`, and `main`'s `load_pretrained_weights`
dropped shape mismatches without comment, so the tokenizer started random. The app's own
pilot script was *more* correct here: it summed the two temporal kernels explicitly.

**2. LoRA never reached the attention layers.** The vendored `AttentionLS` uses `self.qkv`
and `self.proj` (`workshop_infrastructure/models/transformer_ls.py:47-49`). The target list
`[q_proj, v_proj, k_proj, out_proj, fc1, fc2]` — used by both `main`'s default and this
app's config — matched only `fc1`/`fc2`. PEFT raises only when *nothing* matches, so this
was silent, and it affects every result produced before this change.

**3. `feature/fix_lora` already solved most of it.** That branch contained all of `main`
plus `models/weight_adaptation.py`, `discover_head_modules()` + `modules_to_save`,
`strict_shapes`, corrected LoRA targets, `data.pooling`, per-split sample caps and a test
suite. Building on `main` instead would have meant reimplementing all of it.

## Decisions, and why

| Decision | Why |
|---|---|
| Base on `feature/fix_lora`, not `main` | It already contained the weight-adaptation and head-trainability work; starting from `main` would have duplicated it and guaranteed a later conflict |
| **Slice** the pretrained tokenizer to 3 channels rather than keeping the `ChannelAdapter` | The Conv3d(3→13) preserved every pretrained weight but put a *randomly initialized* layer in front of them, so step 0 was not the pretrained function either. Slicing gives the pretrained tokenizer restricted to those channels, with no random layer in the input path at all |
| **Also make the tokenizer trainable** | See below — slicing alone is not sufficient, and this was the substantive open question |
| Port the script and config first; notebooks second | The notebooks would have had to be rewritten twice if the API moved under them |
| Generalize the machine-specific scripts rather than keep or drop them | What they do is useful on any cluster; only the paths were personal |
| Do **not** carry the branch's `environment.yml` pins forward | The current file already pins the same packages deliberately to *newer* versions, and the two the branch added resolve transitively. Folding the old pins in would have been a two-year downgrade to fix a problem that no longer exists |

## The one question that needed an answer: is slicing enough?

Slicing sets the tokenizer's starting point. It does not make the sliced tokenizer a
function the backbone knows how to consume.

`LinearEmbedding` is `patch_embed(x) + pos_embed` with no normalization in between
(`workshop_infrastructure/models/embedding.py:115-127`), and the blocks are pre-norm —
`x = x + drop_path(mlp(norm2(filter(norm1(x)))))` — so LayerNorm normalizes each *branch
input* and never the residual stream. `pos_embed` is a fixed-amplitude Fourier buffer, not a
learned parameter that could rescale alongside the content. So taking 3 channels instead of
13 shrinks the content term while the position term stays put, and no downstream
normalization restores that ratio, because it acts on the sum. A LoRA adapter cannot
substitute either: it acts *after* tokenization and cannot change a per-channel linear map.
The tokenizer is the only layer that can rescale its own output.

**Predicted, then measured.** An independence argument — every channel z-scored and
uncorrelated — gives `sqrt(3/13) = 0.48`. That was written into the plan as an estimate to
check, not a fact. Measured on real normalized data
(`tools/measure_token_scale.py`, 3 samples at 1024px):

| | token std |
|---|---|
| 13-channel pretrained tokenizer | 2.02 – 2.27 |
| 3-channel sliced tokenizer | 1.23 – 1.33 |
| **ratio** | **0.59** (0.584 – 0.607) |

So the effect is real but **milder than predicted** — the channels are correlated and
`aia304/193/171` are not average channels. Position's share of each token grows **1.68x**,
not 2.1x. The direction and the magnitude both still support training the tokenizer, so the
conclusion stood; the number in the documentation was corrected to the measured one.

Cost: `3 x 16 x 16 x 1280 + 1280` = **984,320** parameters, comparable to the ~1.5M the LoRA
adapters themselves add, and 0.27% of the backbone. It trains at the base learning rate, not
the head's multiplier, because it starts from pretrained weights.

It was made a config knob (`model.trainable_backbone_modules`) rather than hardcoded,
precisely so the decision could be tested rather than asserted.

### The A/B: does training the tokenizer actually help?

Both arms identical but for `model.trainable_backbone_modules`, same seed,
`--deterministic warn`, 40 train / 20 val samples, 8 epochs, one A100 each. The arms differ
by exactly 984,320 trainable parameters, which is the tokenizer.

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| tokenizer **trainable** | 1.2911 | 1.2718 | 1.2652 | 1.2623 | 1.2604 | 1.2577 | 1.2547 | **1.2535** |
| tokenizer **frozen** | 1.3012 | 1.2760 | 1.2677 | 1.2641 | 1.2624 | 1.2601 | 1.2586 | **1.2583** |

Training the tokenizer is ahead at **every epoch**, ending 0.0048 lower (0.38%).

**How much this proves: not much on its own.** One seed, 40 training samples, 8 epochs, and a
gap of 0.4% — this is indicative, not conclusive, and it would be wrong to quote the
difference as an effect size. What it does do is rule out the outcome that would have
contradicted the design: the frozen arm is not better, and the ordering is consistent across
all eight epochs rather than crossing over.

The gap is small for a reason worth understanding. The frozen arm is not helpless — its LoRA
adapters and its decoder head still train, and they can partially compensate downstream for a
tokenizer whose output is scaled wrong. What they cannot do is change a per-channel linear
map, which is why the compensation is partial and why the ordering holds.

So the case for the trainable tokenizer rests mainly on the measurement above — the 1.68x
shift in position's share of each token, which is a property of the architecture and not of
any particular run — with the A/B as a consistency check rather than as the evidence. If you
want it to be evidence, run several seeds at a realistic sample count and compare
distributions, not single numbers.

## What was done

1. **Merged** `update/cosmic_caribou` into a branch off `feature/fix_lora`. Three conflicts,
   each with a settled resolution: `helio_aws.py` deleted (superseded by `s3_mode`), the
   template notebook taken from the target, `environment.yml` taken from the target.
   `helio_boto.py`, the `Surya` submodule, the preliminary script and the template's old
   `config.yaml` merged away cleanly.
2. **Extended `adapt_patch_embed_weight()`** to select along the channel axis, composing
   with the existing frame selection into one gather. A channel-count change without
   explicit `channel_indices` raises rather than guessing, because the shapes do not
   determine the mapping.
3. **Added `model.trainable_backbone_modules`**, resolved through
   `resolve_trainable_backbone_modules()` into PEFT's `modules_to_save`. Names must match
   exactly one module, since PEFT matches by suffix.
4. **Moved the Lightning modules into infrastructure** behind a required `target_fn`. The
   second app needing the same loop was the condition for promoting it; the one
   task-specific line was target extraction, which was inlined and wrong for image targets.
5. **Rebuilt the app** on the typed config, the builders and the shared helpers.
6. **Generalized** the conda setup script and the sbatch file.
7. **Documented**: this file, `MIGRATION.md`, `README.md`, and additions to `CLAUDE.md` and
   `ADAPTING.md`.
8. **Rebuilt the three notebooks** from the current template notebooks, in a second pass once
   the script and config were settled. Repairing them in place was considered and rejected: a
   third of their cells were environment scaffolding, and the `ChannelAdapter` and the
   hand-rolled checkpoint loop were superseded designs rather than renamed imports. The
   developer's plots and teaching markdown were ported across; 29 / 37 / 45 code cells became
   14 / 18 / 19, and 5.2 MB of stored outputs were stripped to match the template's
   convention of committing none.

   Two defects surfaced during the rebuild. The notebooks taught **RRSE**, a metric this app
   never computed — inherited prose from the flare template, invisible because prose does not
   fail a test. And plotting revealed that `inverse_transform_data` indexes its per-channel
   statistics positionally by `dataset.channels`, while `ts` and `forecast` hold their own
   channel subsets in their own order; for the shipped config `ts[0]` is `aia304` while
   `channels[0]` is `aia171`. `Euv2MagDataset.inverse_transform_subset()` now does the
   scatter-by-name, and the training script's figure uses it too rather than keeping a second
   copy.

## What was verified

| Check | Result |
|---|---|
| No dead imports (`helio_boto`, `helio_aws`, `Surya`) | clean |
| Test suite | 103 passed, including new channel-selection, LoRA-trainability and config tests |
| Tokenizer equals the pretrained slice | exact, verified tensor-by-tensor against the checkpoint |
| Pretrained tensors loaded | 156/159 — the 2 missing are the pretraining decoder, plus the regenerated `pos_embed` buffer; 3 adapted, none dropped |
| Attention adapted | `attn.qkv` and `attn.proj` in all 8 attention blocks |
| Trainable set | tokenizer 984,320 + LoRA 1,515,520 + head 327,936 = 2,827,776 (1.42%) |
| Tokenizer optimizer group | backbone group at 1e-4, not the head's 1e-3 |
| End-to-end run | finite `val_loss`, checkpoint written, figure in physical units (Gauss) |
| Determinism (`--deterministic warn`) | `val_loss` 1.836134 on three independent runs |
| Baseline path | trains, `val_loss` 2.789744 |
| DDP on 2 GPUs | both ranks report identical `val_loss`, so `sync_dist` reduces correctly |
| Linear-probe regime (`use_lora: false`, `freeze_backbone: true`) | 1,312,256 trainable = head 327,936 + tokenizer 984,320, rest of backbone frozen |

A code review over the finished branch found six defects, all fixed before merge and worth
recording because most were silent rather than loud:

| Defect | Consequence |
|---|---|
| `trainable_backbone_modules` passed only inside the `use_lora` branch | the linear-probe regime froze the channel-sliced tokenizer — precisely what the design forbids |
| `resolve_trainable_backbone_modules` returned the as-written suffix | `model.get_submodule()` resolves from the top-level model, so the above fix crashed until it returned the fully-qualified name |
| `ckpt_in_chans` silently defaulted to `in_chans` | omitting it alongside `channel_indices` passes every check and gathers the wrong planes |
| `HelioSpectformer2D.from_config` never read `ft_unembedding_type` | `perceiver` was a silent no-op for the reference 2D model |
| `#SBATCH --output` into `slurm_logs/` with the `mkdir` in the job body | SLURM opens the file before the body runs, so the first submission fails |
| `conda config --remove-key envs_dirs` in the setup script | wipes entries the script never added, breaking a user's other environments |

The README also described the task as co-temporal while the config predicts +60 min; the
documentation was corrected to the config rather than the reverse, and the choice flagged.

## What was deliberately not done

- **No notebook execution test was added.** This repo has no notebook CI, and adding it would
  mean a new dependency plus test infrastructure that needs a GPU and cached data to be
  meaningful. The notebooks were each executed once by hand instead, and what keeps them
  honest is the same discipline the template relies on: every literal reads from `cfg`, so
  the notebook and the script cannot numerically disagree even though nothing checks.
- **The `environment.yml` pins were not folded in**, contrary to the original plan. Comparing
  the actual versions showed the target's pins were newer and deliberate; the plan had been
  written before that comparison. Recorded here because a plan that was wrong on the facts
  is worth more as a record than a plan quietly edited to look right.
