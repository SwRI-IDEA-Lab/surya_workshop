#!/usr/bin/env python3
"""
Surya patch-level embedding analysis.

Extracts per-patch backbone embeddings (no global average pooling),
projects them to 2D with UMAP, and saves diagnostic figures.

Each data point in the embedding volume is a single spatial patch token,
labelled by its parent sample's GOES class and its (row, col) position
on the solar disk.

Figures saved to OUTPUT_DIR:
  umap_by_label.png       -- patch embeddings coloured by GOES class
  umap_by_position.png    -- patch embeddings coloured by solar-disk position
  spatial_emb_magnitude.png -- per-patch-position mean embedding L2 norm
  silhouette.png          -- per-class mean silhouette score
  embedding_index.csv     -- the exact samples used (for reproducibility)
  patch_embeddings.npy    -- (N_samples, N_patches_kept, embed_dim)
  patch_positions.npy     -- (N_samples, N_patches_kept, 2)  [row, col]
  timestamps.npy          -- per-sample timestamps
"""

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm.auto import tqdm

# ── locate repo root ────────────────────────────────────────────────────────
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir
while not (_repo_root / "Surya").exists() and _repo_root != _repo_root.parent:
    _repo_root = _repo_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from workshop_infrastructure.datasets.helio import HelioNetCDFDataset
from workshop_infrastructure.models.helio_spectformer import HelioSpectFormer
from workshop_infrastructure.utils import load_pretrained_weights, build_scalers


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INDEX_PATH = "/nobackupnfs1/sroy14/processed_data/Helio/csv_files/full_data_201006_to_202412_with_priority.csv"
SCALERS_PATH    = str(_repo_root / "downstream_apps/template/assets/scalers.yaml")
CHECKPOINT_PATH = str(_repo_root / "downstream_apps/template/assets/surya.366m.v1.pt")

LABEL_FILE_PATH      = str(_repo_root / "downstream_apps/template/data/hek_flare_catalog.csv")
LABEL_TIME_COL       = "start_time"
LABEL_VALUE_COL      = "GOES_class"
LABEL_TIME_TOLERANCE = pd.Timedelta("6h")

START_DATE  = "2014-01-01"
END_DATE    = "2014-12-31"
MAX_SAMPLES = 200          # number of SDO observations to embed
# Patches per observation to keep for UMAP (65 536 total; subsampled for tractability)
MAX_PATCHES_PER_SAMPLE = 512
RANDOM_SEED = 42

PATCH_GRID = 256           # 4096 / 16 = 256 patches per spatial axis

CHANNELS = [
    "aia94", "aia131", "aia171", "aia193", "aia211",
    "aia304", "aia335", "aia1600",
    "hmi_m", "hmi_bx", "hmi_by", "hmi_bz", "hmi_v",
]
MODEL_CONFIG = dict(
    img_size          = 4096,
    patch_size        = 16,
    in_chans          = 13,
    embed_dim         = 1280,
    time_embedding    = {"type": "linear", "time_dim": 1},
    depth             = 10,
    n_spectral_blocks = 2,
    num_heads         = 16,
    mlp_ratio         = 4.0,
    drop_rate         = 0.0,
    window_size       = 2,
    dp_rank           = 4,
    nglo              = 1,
    checkpoint_layers = list(range(10)),
    finetune          = True,
)

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1

OUTPUT_DIR = Path(__file__).parent / "outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-date",  default=START_DATE)
    p.add_argument("--end-date",    default=END_DATE)
    p.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    p.add_argument("--max-patches", type=int, default=MAX_PATCHES_PER_SAMPLE,
                   help="Patches kept per observation for UMAP (default: %(default)s)")
    p.add_argument("--seed",        type=int, default=RANDOM_SEED)
    p.add_argument("--output-dir",  type=Path, default=OUTPUT_DIR)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def subsample_index(index_df: pd.DataFrame, start: str, end: str,
                    max_samples: int) -> pd.DataFrame:
    """Temporally uniform subsample of the index within [start, end]."""
    mask = (
        (index_df["timestep"] >= pd.Timestamp(start))
        & (index_df["timestep"] <= pd.Timestamp(end))
    )
    window = index_df[mask].reset_index(drop=True)
    if len(window) > max_samples:
        step = len(window) // max_samples
        window = window.iloc[::step].head(max_samples).reset_index(drop=True)
    return window


def align_labels(timestamps: pd.DatetimeIndex,
                 label_file: str,
                 time_col: str,
                 value_col: str,
                 tolerance: pd.Timedelta) -> tuple[np.ndarray, np.ndarray]:
    """
    Nearest-timestamp join between *timestamps* and the label CSV.

    Returns
    -------
    matched_indices : int array, indices into *timestamps* that were matched
    labels          : string array of GOES class letters (e.g. 'M')
    """
    label_df = pd.read_csv(label_file)
    label_df[time_col] = pd.to_datetime(label_df[time_col], utc=False)
    label_df = label_df.sort_values(time_col).reset_index(drop=True)

    samples_df = pd.DataFrame({"timestep": timestamps, "orig_idx": range(len(timestamps))})
    samples_df = samples_df.sort_values("timestep").reset_index(drop=True)

    joined = pd.merge_asof(
        samples_df,
        label_df[[time_col, value_col]].rename(columns={time_col: "timestep"}),
        on        = "timestep",
        direction = "nearest",
        tolerance = tolerance,
    )
    matched = joined[value_col].notna()
    matched_indices = joined.loc[matched, "orig_idx"].values
    raw_labels      = joined.loc[matched, value_col].values
    labels = np.array([str(v)[0].upper() for v in raw_labels])
    return matched_indices, labels


# ---------------------------------------------------------------------------
# Backbone helpers
# ---------------------------------------------------------------------------
def build_backbone(checkpoint: str, device: torch.device) -> torch.nn.Module:
    backbone = HelioSpectFormer(**MODEL_CONFIG)
    load_pretrained_weights(backbone, checkpoint)
    backbone.eval().to(device)
    n = sum(p.numel() for p in backbone.parameters())
    print(f"Backbone: {n / 1e6:.1f}M parameters  ({device})")
    return backbone


@torch.no_grad()
def extract_patch_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_patches_keep: int,
    patch_grid: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract per-patch embeddings without any pooling.

    The backbone returns tokens of shape (B, 1 + N_spatial, embed_dim) when
    nglo=1.  Token 0 is the learned global token; tokens 1: are the spatial
    patch tokens laid out in raster order (row-major, top-left origin).

    We draw *n_patches_keep* patch indices (same set for every sample so that
    spatial position is comparable across samples) and return:

    patch_embs  : (N_samples, n_patches_keep, embed_dim)  float32
    patch_pos   : (N_samples, n_patches_keep, 2)           int32  [row, col]
    """
    n_spatial = patch_grid * patch_grid          # 65 536
    keep_idx  = rng.choice(n_spatial, size=min(n_patches_keep, n_spatial), replace=False)
    keep_idx  = np.sort(keep_idx)                # reproducible, spatial order

    rows = (keep_idx // patch_grid).astype(np.int32)
    cols = (keep_idx  % patch_grid).astype(np.int32)
    patch_pos_template = np.stack([rows, cols], axis=1)  # (n_keep, 2)

    model.eval()
    all_embs = []

    for batch in tqdm(dataloader, desc="Extracting patch embeddings"):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        tokens = model(batch)                    # (B, 1 + N_spatial, embed_dim)
        spatial_tokens = tokens[:, 1:, :]        # drop global token → (B, N_spatial, D)
        kept = spatial_tokens[:, keep_idx, :]    # (B, n_keep, D)
        all_embs.append(kept.cpu().float().numpy())

    patch_embs = np.concatenate(all_embs, axis=0)  # (N_samples, n_keep, D)
    patch_pos  = np.broadcast_to(
        patch_pos_template[None], (patch_embs.shape[0], *patch_pos_template.shape)
    ).copy()

    return patch_embs, patch_pos


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
GOES_PALETTE = {
    "A": "#aec6cf", "B": "#90ee90", "C": "#ffd700",
    "M": "#ff8c00", "X": "#dc143c",
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def plot_umap_by_label(emb2d: np.ndarray, patch_labels: np.ndarray,
                       label_order: list[str], out_path: Path,
                       title_suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for cls in label_order:
        m = patch_labels == cls
        ax.scatter(emb2d[m, 0], emb2d[m, 1],
                   c=GOES_PALETTE.get(cls, "grey"),
                   label=f"GOES {cls}  (n={m.sum():,})",
                   s=4, alpha=0.5, linewidths=0)
    ax.legend(title="Flare class", loc="best", markerscale=3)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Surya patch embeddings – UMAP by label{title_suffix}")
    _save(fig, out_path)


def plot_umap_by_position(emb2d: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                          patch_grid: int, out_path: Path,
                          title_suffix: str = "") -> None:
    # encode (row, col) as hue/luminance in HSV so spatial proximity → colour similarity
    hue = cols / patch_grid                # 0 (left) → 1 (right)
    val = 1.0 - rows / patch_grid          # 1 (top)  → 0 (bottom)
    sat = np.ones_like(hue) * 0.8
    hsv = np.stack([hue, sat, val], axis=1)
    rgba = mcolors.hsv_to_rgb(hsv)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(emb2d[:, 0], emb2d[:, 1],
               c=rgba, s=4, alpha=0.5, linewidths=0)
    # colour-wheel legend approximation via two colour bars
    sm_h = plt.cm.ScalarMappable(cmap="hsv",
                                  norm=mcolors.Normalize(0, patch_grid))
    sm_h.set_array([])
    sm_v = plt.cm.ScalarMappable(cmap="gray_r",
                                  norm=mcolors.Normalize(0, patch_grid))
    sm_v.set_array([])
    plt.colorbar(sm_h, ax=ax, fraction=0.03, pad=0.01).set_label("patch column →")
    plt.colorbar(sm_v, ax=ax, fraction=0.03, pad=0.06).set_label("patch row ↓")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Surya patch embeddings – UMAP by spatial position{title_suffix}")
    _save(fig, out_path)


def plot_spatial_magnitude(patch_embs: np.ndarray, patch_pos: np.ndarray,
                           patch_grid: int, out_path: Path,
                           title_suffix: str = "") -> None:
    """
    Spatial heatmap: for each patch position, mean L2 norm of its embedding
    vector across all samples.  Reveals which solar-disk regions produce the
    most energetic (distinctive) representations.
    """
    # patch_embs: (N_samples, n_keep, D)
    # patch_pos:  (N_samples, n_keep, 2)  — same positions every sample
    norms = np.linalg.norm(patch_embs, axis=-1)   # (N_samples, n_keep)
    mean_norms = norms.mean(axis=0)                # (n_keep,)

    grid = np.full((patch_grid, patch_grid), np.nan)
    pos = patch_pos[0]                             # same for all samples
    for i, (r, c) in enumerate(pos):
        grid[r, c] = mean_norms[i]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, origin="upper", cmap="inferno",
                   vmin=np.nanpercentile(grid, 2),
                   vmax=np.nanpercentile(grid, 98))
    plt.colorbar(im, ax=ax).set_label("Mean embedding L2 norm")
    ax.set_xlabel("Patch column")
    ax.set_ylabel("Patch row")
    ax.set_title(f"Spatial map of mean embedding magnitude{title_suffix}")
    _save(fig, out_path)


def plot_silhouette(matched_embs: np.ndarray, labels: np.ndarray,
                    out_path: Path) -> None:
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.preprocessing import LabelEncoder

    if len(np.unique(labels)) < 2:
        print("  silhouette: need ≥2 classes — skipping")
        return

    le = LabelEncoder()
    labels_int = le.fit_transform(labels)
    score = silhouette_score(matched_embs, labels_int, metric="cosine")
    sample_scores = silhouette_samples(matched_embs, labels_int, metric="cosine")
    print(f"  silhouette score (cosine, {len(le.classes_)} classes): {score:.4f}")

    cls_means = [sample_scores[labels_int == i].mean() for i in range(len(le.classes_))]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(le.classes_, cls_means,
           color=[GOES_PALETTE.get(c, "steelblue") for c in le.classes_])
    ax.axhline(score, color="red", linestyle="--", label=f"Overall: {score:.3f}")
    ax.set_xlabel("GOES class")
    ax.set_ylabel("Mean silhouette")
    ax.set_title("Silhouette score by GOES class\n(cosine distance on full-dim embeddings)")
    ax.legend()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Device  : {DEVICE}")
    print(f"Output  : {args.output_dir}")

    # ── 1. Index ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading index …")
    index_df = pd.read_csv(INDEX_PATH)
    index_df["timestep"] = pd.to_datetime(index_df["timestep"])
    index_df = index_df[index_df["present"] == 1].reset_index(drop=True)
    print(f"  {len(index_df):,} valid samples  "
          f"({index_df['timestep'].min().date()} → {index_df['timestep'].max().date()})")

    embed_df = subsample_index(index_df, args.start_date, args.end_date, args.max_samples)
    embed_index_path = args.output_dir / "embedding_index.csv"
    embed_df.to_csv(embed_index_path, index=False)
    print(f"  {len(embed_df)} samples selected  →  {embed_index_path}")

    # ── 2. Backbone ───────────────────────────────────────────────────────
    print("\n[2/6] Building backbone …")
    backbone = build_backbone(CHECKPOINT_PATH, DEVICE)

    # ── 3. Dataset / DataLoader ───────────────────────────────────────────
    print("\n[3/6] Building dataset …")
    scalers = build_scalers(SCALERS_PATH)
    dataset = HelioNetCDFDataset(
        index_path               = str(embed_index_path),
        time_delta_input_minutes = [0],
        time_delta_target_minutes= 60,
        n_input_timestamps       = 1,
        rollout_steps            = 0,
        scalers                  = scalers,
        channels                 = CHANNELS,
        phase                    = "embed",
        load_forecast_frames     = False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = DEVICE.type == "cuda",
    )
    timestamps = pd.to_datetime([str(ts) for ts in dataset.valid_indices])
    print(f"  {len(dataset)} samples  "
          f"({timestamps[0].date()} → {timestamps[-1].date()})")

    # ── 4. Extract patch embeddings ───────────────────────────────────────
    print(f"\n[4/6] Extracting patch embeddings "
          f"({args.max_patches} patches/sample) …")
    patch_embs, patch_pos = extract_patch_embeddings(
        backbone, loader, DEVICE, args.max_patches, PATCH_GRID, rng
    )
    # patch_embs : (N_samples, n_keep, embed_dim)
    # patch_pos  : (N_samples, n_keep, 2)
    np.save(args.output_dir / "patch_embeddings.npy", patch_embs)
    np.save(args.output_dir / "patch_positions.npy",  patch_pos)
    np.save(args.output_dir / "timestamps.npy",       timestamps.values)
    print(f"  patch_embs shape: {patch_embs.shape}")

    # ── 5. Labels ─────────────────────────────────────────────────────────
    print("\n[5/6] Aligning labels …")
    matched_sample_idx, sample_labels = align_labels(
        timestamps, LABEL_FILE_PATH, LABEL_TIME_COL,
        LABEL_VALUE_COL, LABEL_TIME_TOLERANCE,
    )
    print(f"  matched {len(matched_sample_idx)} / {len(timestamps)} samples")
    label_order = [c for c in ["A", "B", "C", "M", "X"]
                   if c in np.unique(sample_labels)]
    for cls in label_order:
        print(f"    GOES {cls}: {(sample_labels == cls).sum()}")

    # Select matched samples and replicate each sample's label across its patches
    matched_patch_embs = patch_embs[matched_sample_idx]   # (N_m, n_keep, D)
    matched_patch_pos  = patch_pos[matched_sample_idx]    # (N_m, n_keep, 2)
    patch_labels = np.repeat(sample_labels, matched_patch_embs.shape[1])  # (N_m*n_keep,)

    # Flatten patches for UMAP
    N_m, n_keep, D = matched_patch_embs.shape
    flat_embs = matched_patch_embs.reshape(-1, D)                # (N_m*n_keep, D)
    flat_rows  = matched_patch_pos[:, :, 0].reshape(-1)          # (N_m*n_keep,)
    flat_cols  = matched_patch_pos[:, :, 1].reshape(-1)          # (N_m*n_keep,)

    print(f"  total patch vectors for UMAP: {flat_embs.shape[0]:,}")

    # ── 6. UMAP + figures ─────────────────────────────────────────────────
    print("\n[6/6] Running UMAP and saving figures …")
    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError("umap-learn is required: pip install umap-learn")

    reducer = umap_lib.UMAP(
        n_neighbors  = UMAP_N_NEIGHBORS,
        min_dist     = UMAP_MIN_DIST,
        n_components = 2,
        random_state = args.seed,
        verbose      = True,
    )
    emb2d = reducer.fit_transform(flat_embs)
    np.save(args.output_dir / "patch_embedding_2d.npy", emb2d)
    print(f"  2D projection: {emb2d.shape}")

    title_suffix = (
        f"\n{args.start_date} → {args.end_date}  "
        f"({N_m} samples × {n_keep} patches = {flat_embs.shape[0]:,} points)"
    )

    plot_umap_by_label(
        emb2d, patch_labels, label_order,
        args.output_dir / "umap_by_label.png",
        title_suffix,
    )
    plot_umap_by_position(
        emb2d, flat_rows, flat_cols, PATCH_GRID,
        args.output_dir / "umap_by_position.png",
        title_suffix,
    )
    # Spatial magnitude uses all samples (not only matched) for better statistics
    plot_spatial_magnitude(
        patch_embs, patch_pos, PATCH_GRID,
        args.output_dir / "spatial_emb_magnitude.png",
        f"\n({len(patch_embs)} samples)",
    )
    # Silhouette on per-sample mean embeddings (patch-mean per sample = compact representation)
    sample_mean_embs = matched_patch_embs.mean(axis=1)   # (N_m, D)
    plot_silhouette(
        sample_mean_embs, sample_labels,
        args.output_dir / "silhouette.png",
    )

    print(f"\nDone.  All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
