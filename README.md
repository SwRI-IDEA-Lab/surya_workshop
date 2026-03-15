# Surya Workshop

A template repository for fine-tuning [Surya](https://github.com/NASA-IMPACT/Surya), the first foundation model for heliophysics, on your own downstream solar science tasks.

---

## The Surya Foundation Model

**Surya** is a 366-million-parameter spatiotemporal transformer pre-trained on full-resolution data from NASA's [Solar Dynamics Observatory (SDO)](https://sdo.gsfc.nasa.gov/). It was developed as a NASA-IMPACT / IBM AI4Science collaboration and is described in:

> *Surya: A Foundation Model for Heliophysics* — [arXiv:2508.14112](https://arxiv.org/abs/2508.14112)

The model ingests 13-channel SDO image stacks (8 AIA wavelengths + 4 HMI magnetic components + HMI doppler velocity) at native 4096×4096 resolution and has demonstrated strong performance across a range of solar physics tasks:

| Task | Improvement over prior state-of-the-art |
|---|---|
| Solar flare forecasting (TSS) | +22% |
| Solar wind speed prediction (RMSE) | +19% |
| Active region segmentation | — |
| EUV spectra modeling (1,343 bands) | — |

**Resources**
- Model weights: [`nasa-ibm-ai4science/Surya-1.0`](https://huggingface.co/nasa-ibm-ai4science/Surya-1.0) on HuggingFace
- Pre-training dataset: [`nasa-ibm-ai4science/core-sdo`](https://huggingface.co/datasets/nasa-ibm-ai4science/core-sdo) on HuggingFace
- Source code: [NASA-IMPACT/Surya](https://github.com/NASA-IMPACT/Surya)
- License: Apache 2.0

### Architecture

Surya uses two novel transformer block types that make it efficient on full-resolution solar imagery:

- **Spectral Gating** — transforms patches to the frequency domain via FFT, applies learnable complex weights, then returns via iFFT. Captures global structure efficiently.
- **Long-Short Attention** — combines local windowed attention (`window_size=2`) with global attention via dynamic projection (`dp_rank=4`). Handles the 4096×4096 spatial extent without quadratic cost.

The full backbone is 2 spectral gating blocks followed by 8 long-short attention blocks, with patch size 16 and embedding dimension 1280.

---

## Purpose of This Repository

Surya is a powerful foundation, but foundation models only create scientific value when researchers can adapt them to their own questions. That adaptation step — loading pre-trained weights, defining a task-specific head, wiring up data and metrics, and running a reproducible training loop requires support when executed for the first time.

This repository provides a **clean, heavily documented template** that addresses that need. The goal is to provide a well-explained starting point that a scientist can read, understand, and modify in an afternoon:

- Every component has a clear home and a documented interface.
- A single YAML file controls all hyperparameters.
- Numbered notebooks walk through each stage of the workflow interactively before the production script ties them together.
- The template task (solar flare intensity regression) is realistic enough to illustrate the full pattern, but simple enough that it doesn't obscure what you need to change.

---

## Repository Structure

```
surya_workshop/
│
├── data/
│   └── indices/                        # Pre-built CSV index files for the SDO dataset
│       ├── surya_aws_s3_full_index.csv # Complete index of all available SDO timesteps on S3
│       ├── surya_aws_s3_train.csv      # Training split
│       ├── surya_aws_s3_val.csv        # Validation split
│       └── surya_aws_s3_test.csv       # Test split
│
├── workshop_infrastructure/            # Shared utilities used by all downstream apps
│   ├── configs.py                      # Typed dataclasses: ModelConfig, LoraAdapterConfig, TimeEmbeddingConfig
│   ├── utils.py                        # build_scalers(), apply_peft_lora(), load_pretrained_weights(),
│   │                                   # UploadBestCheckpointToS3, create_logger
│   ├── benchmark_s3.py                 # Benchmark S3 download throughput to tune transfer settings
│   ├── datasets/
│   │   ├── helio.py                    # HelioNetCDFDataset — base dataset (local + S3, signum-log normalization)
│   │   └── transformations.py          # Additional data transformations
│   ├── models/
│   │   ├── finetune_models.py          # HelioSpectformer1D / HelioSpectformer2D fine-tuning wrappers
│   │   ├── helio_spectformer.py        # Full backbone (HelioSpectFormer)
│   │   ├── spectformer.py              # Spectral gating blocks
│   │   ├── transformer_ls.py           # Long-short attention blocks
│   │   ├── embedding.py                # Temporal embedding modules
│   │   └── flow.py                     # Learned flow utilities
│   └── data/
│       ├── create_csv_index.py         # Build a timestep CSV index from a collection of NetCDF files
│       └── split_csv_index.py          # Split an index into train / val / test sets
│
└── downstream_apps/
    └── template/                       # Solar flare intensity regression (the working template task)
        │
        ├── ADAPTING.md                 # Step-by-step guide for creating your own downstream app
        │
        ├── configs/
        │   └── config_script.yaml      # Single source of truth for all hyperparameters
        ├── configs.py                  # DataConfig, TrainingConfig, OutputConfig + load_config()
        │
        ├── assets/
        │   ├── scalers.yaml            # Per-channel normalization statistics (downloaded on first run)
        │   └── surya.366m.v1.pt        # Pre-trained Surya weights (downloaded on first run)
        ├── data/
        │   └── hek_flare_catalog.csv   # HEK flare event catalog used as regression targets
        │
        ├── datasets/
        │   └── template_dataset.py     # FlareDSDataset — extends HelioNetCDFDataset with flare labels
        ├── lightning_modules/
        │   └── pl_simple_baseline.py   # FlareLightningModule — Lightning training + validation loop
        ├── metrics/
        │   └── template_metrics.py     # FlareMetrics — MSE loss + RRSE evaluation metrics
        ├── models/
        │   └── simple_baseline.py      # RegressionFlareModel — linear baseline (no backbone)
        │
        ├── download_scalers_and_weights.sh   # Download assets/ from HuggingFace (run once)
        │
        ├── 0_dataset_dataloader_template.ipynb   # Step 1: explore the dataset and DataLoader
        ├── 1_baseline_template.ipynb             # Step 2: train a linear baseline
        ├── 2_finetune_template_1D.ipynb          # Step 3: fine-tune Surya interactively
        └── 3_finetune_template_1D.py             # Step 4: production training script
```

**Two layers to understand:**

- `workshop_infrastructure/` — reusable components that any downstream app can import. You should rarely need to change anything here.
- `downstream_apps/template/` — everything specific to one task. When you build your own app, you copy this folder and modify it.

---

## How to Use This Repository

### 1. Environment setup

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/your-org/surya_workshop.git
cd surya_workshop

# Create and activate the conda environment
conda env create -f environment.yml
conda activate surya_ws
```

Python 3.12+ is required. Key dependencies: PyTorch, PyTorch Lightning, PEFT, WandB, SunPy, xarray, Dask, fsspec.

### 2. Work through the template notebooks in order

Each notebook is self-contained and builds directly on the previous one. They are designed to be run interactively so you can inspect data, check tensor shapes, and verify each component before committing to a full training run.

| Notebook | What it teaches |
|---|---|
| `0_dataset_dataloader_template.ipynb` | How SDO data is indexed, loaded, and normalized; what a sample dict looks like |
| `1_baseline_template.ipynb` | Training a simple linear model end-to-end; defines the metric and evaluation baseline |
| `2_finetune_template_1D.ipynb` | Loading Surya weights, applying LoRA, and fine-tuning interactively |

### 3. Run the production training script

Once you're satisfied with the notebook workflow, `3_finetune_template_1D.py` runs the same logic as notebook 2 but with multi-GPU DDP support, checkpoint saving, and WandB logging:

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python -m downstream_apps.template.3_finetune_template_1D \
    --config downstream_apps/template/configs/config_script.yaml

# Multi-GPU (DDP)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m downstream_apps.template.3_finetune_template_1D \
    --config downstream_apps/template/configs/config_script.yaml

# Quick sanity check (cap epochs without editing the YAML)
CUDA_VISIBLE_DEVICES=0 python -m downstream_apps.template.3_finetune_template_1D \
    --config downstream_apps/template/configs/config_script.yaml \
    --max-epochs 2 --no-wandb
```

**All hyperparameters live in `config_script.yaml`** — batch size, learning rate, LoRA settings, S3 paths, and more. The script reads the YAML via `load_config()` and returns a fully typed `TrainingConfig`, so IDE autocompletion works and typos are caught at startup rather than mid-training.

Set `max_samples: 10` in the YAML during development to cap the dataset size for fast iteration.

### 4. (Optional) Tune S3 download performance

If your training data is read from S3, `workshop_infrastructure/benchmark_s3.py` measures download throughput across combinations of thread concurrency and part size and recommends the best settings for your connection:

```bash
python -m workshop_infrastructure.benchmark_s3 s3://bucket/path/to/file.nc --anon --quick
```

One of the files in the surya index works fine (e.g. `s3://nasa-surya-bench/2011/01/20110131_0000.nc`).  The `--quick` flag runs a 9-cell grid and finishes in about 2–3 minutes. Copy the recommended values into the `data:` section of `config_script.yaml`:

```yaml
s3_boto3_max_concurrency: 8   # suggested by benchmark
s3_boto3_part_size_mb: 32     # suggested by benchmark
```

On EC2 in the same AWS region as the bucket, expect 500–1000+ MB/s. Over a regular internet connection, 20–150 MB/s is typical — in either case the benchmark will find the fastest achievable settings.

> **EC2 users:** to ensure S3 traffic routes over the AWS internal backbone and never touches an internet or NAT gateway, confirm that a **VPC S3 Gateway Endpoint** is attached to your VPC (AWS Console → VPC → Endpoints → filter by "S3 Gateway"). It is free and takes two minutes to create. Without it, even same-region traffic passes through a gateway, reducing throughput and incurring data-transfer costs. The benchmark script will remind you of this automatically when it detects it is running on EC2.

### 5. Adapt the template for your own task

Read [`downstream_apps/template/ADAPTING.md`](downstream_apps/template/ADAPTING.md) for a step-by-step guide. The short version:

1. `cp -r downstream_apps/template downstream_apps/your_task`
2. Edit `datasets/template_dataset.py` to load your labels alongside the SDO image stack.
3. Edit `metrics/template_metrics.py` to define your loss and evaluation metrics.
4. Edit `configs/config_script.yaml` to point at your data and set your hyperparameters.
5. Run notebook 2 → verify the forward pass → run the training script.

You typically do not need to touch `workshop_infrastructure/` at all.

<!-- --- -->

<!-- ## Key Design Decisions

**YAML as single source of truth.** All parameters are declared once in `config_script.yaml` and nowhere else. The CLI exposes only four arguments: `--config` (required), `--no-wandb` (dev toggle), `--train_baseline` (mode switch), and `--max-epochs` (sweep override). This keeps experiment management simple and reproducible.

**Typed configuration.** `load_config()` parses the YAML into a `TrainingConfig` dataclass. Downstream code receives a typed object instead of a raw dict — adding a new field requires editing the dataclass and YAML only, not `load_config()` itself.

**Notebooks and script are parallel, not redundant.** The notebooks are the learning path — they expose internals and make it easy to inspect intermediate results. The script is the production path — it adds DDP, robust checkpointing, and WandB integration. Both read the same YAML.

**LoRA for efficient fine-tuning.** By default, PEFT LoRA adapters are added to all attention and feed-forward layers (rank 8, alpha 8, dropout 0.1). This allows the full Surya backbone to remain effectively frozen while adapting it to a new task with a small number of trainable parameters. LoRA can be disabled in the YAML if you prefer full fine-tuning or backbone freezing. -->
