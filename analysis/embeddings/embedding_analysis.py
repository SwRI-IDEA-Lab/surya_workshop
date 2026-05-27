#!/usr/bin/env python3
"""
Surya patch-level embedding analysis.

Extracts per-patch backbone embeddings (no global average pooling),
projects them to 2D with UMAP, and saves diagnostic figures.

Each data point is a single spatial patch token labelled by the
median segmentation class in the corresponding 16×16 pixel tile
of the 4096×4096 feature mask.

Mask classes:
  -1  NA (off-disk)
   0  Coronal Hole
   1  Quiet Sun
   2  Active Region

Figures saved to OUTPUT_DIR:
  umap_by_mask_class.png      -- patch embeddings coloured by mask class
  umap_by_position.png        -- patch embeddings coloured by solar-disk position
  umap_by_cluster.png         -- patch embeddings coloured by HDBSCAN cluster
  spatial_emb_magnitude.png   -- per-patch-position mean embedding L2 norm
  spatial_mask_labels.png     -- per-patch-position median mask class across all samples
  spatial_cluster_labels.png  -- per-patch-position dominant HDBSCAN cluster
  silhouette.png               -- per-class mean silhouette score (patch level)
  embedding_index.csv          -- the exact samples used (for reproducibility)
  patch_embeddings.npy         -- (N_samples, N_patches_kept, embed_dim)
  patch_positions.npy          -- (N_samples, N_patches_kept, 2)  [row, col]
  patch_mask_labels.npy        -- (N_samples, N_patches_kept)  integer class per patch
  patch_cluster_labels.npy    -- (N_samples * N_patches_kept,) HDBSCAN cluster per point
  timestamps.npy               -- per-sample timestamps
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
while not (_repo_root / "workshop_infrastructure").exists() and _repo_root != _repo_root.parent:
    _repo_root = _repo_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from workshop_infrastructure.datasets.helio import HelioNetCDFDataset
from workshop_infrastructure.models.helio_spectformer import HelioSpectFormer
from workshop_infrastructure.utils import load_pretrained_weights, build_scalers


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INDEX_PATH           = "/nobackupnfs1/sroy14/processed_data/Helio/csv_files/full_data_201006_to_202412_with_priority.csv"
MASKS_DIR            = Path("/nobackupnfs1/sroy14/processed_data/Helio/segment_vu")
MASKS_TIME_TOLERANCE = pd.Timedelta("6min")
MASK_KEY             = "feature_mask"

SCALERS_PATH    = str(_repo_root / "downstream_apps/template/assets/scalers.yaml")
CHECKPOINT_PATH = "/nobackupp17/amunozja/surya.366m.v1.pt"
CACHE_PATH = "/nobackupp17/amunozja/surya_ws_cache"

START_DATE  = "2013-01-01"
END_DATE    = "2020-12-31"
MAX_SAMPLES = 3
MAX_PATCHES_PER_SAMPLE = 512*8*4*4
RANDOM_SEED = 42

PATCH_SIZE = 16
PATCH_GRID = 256           # 4096 / 16

CHANNELS = [
    "aia94", "aia131", "aia171", "aia193", "aia211",
    "aia304", "aia335", "aia1600",
    "hmi_m", "hmi_bx", "hmi_by", "hmi_bz", "hmi_v",
]
MODEL_CONFIG = dict(
    img_size          = 4096,
    patch_size        = PATCH_SIZE,
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
    nglo              = 0,
    checkpoint_layers = list(range(10)),
    finetune          = True,
)

NORMALIZE_EMBEDDINGS = False   # L2-normalise each patch embedding to unit magnitude

UMAP_N_NEIGHBORS = 200
UMAP_MIN_DIST    = 0.99

HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 2

OUTPUT_DIR = Path(__file__).parent / "outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MASK_CLASS_NAMES = {-1: "NA", 0: "Coronal Hole", 1: "Quiet Sun", 2: "Active Region"}
MASK_PALETTE     = {-1: "#aaaaaa", 0: "#4a90d9", 1: "#f5a623", 2: "#d0021b"}


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
    p.add_argument("--masks-dir",   type=Path, default=MASKS_DIR,
                   help="Directory with *_with_masks.npz files (default: %(default)s)")
    p.add_argument("--normalize-embeddings", action="store_true",
                   default=NORMALIZE_EMBEDDINGS,
                   help="L2-normalise each patch embedding to unit magnitude before UMAP/HDBSCAN")
    p.add_argument("--hdbscan-min-cluster-size", type=int,
                   default=HDBSCAN_MIN_CLUSTER_SIZE,
                   help="HDBSCAN min_cluster_size (default: %(default)s)")
    p.add_argument("--hdbscan-min-samples", type=int, default=HDBSCAN_MIN_SAMPLES,
                   help="HDBSCAN min_samples; defaults to min-cluster-size when omitted")
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


def load_mask_timestamps(masks_dir: Path) -> pd.DataFrame:
    """Return a DataFrame with columns [timestep, mask_path] for every *_with_masks.npz file."""
    records = []
    for p in sorted(masks_dir.glob("*_with_masks.npz")):
        date_part = p.stem.split("_with_masks")[0]
        try:
            ts = pd.to_datetime(date_part, format="%Y%m%d_%H%M")
            records.append({"timestep": ts, "mask_path": str(p)})
        except ValueError:
            warnings.warn(f"Could not parse timestamp from {p.name}")
    return pd.DataFrame(records, columns=["timestep", "mask_path"])


def intersect_with_masks(index_df: pd.DataFrame, mask_df: pd.DataFrame,
                          tolerance: pd.Timedelta) -> pd.DataFrame:
    """Return index rows whose timestep has a matching mask within tolerance, with mask_path attached."""
    index_sorted = index_df.sort_values("timestep").reset_index(drop=True)
    mask_sorted  = mask_df[["timestep", "mask_path"]].sort_values("timestep").reset_index(drop=True)
    joined = pd.merge_asof(
        index_sorted, mask_sorted,
        on="timestep", direction="nearest", tolerance=tolerance,
    )
    matched = joined["mask_path"].notna()
    return joined[matched].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Mask label helpers
# ---------------------------------------------------------------------------
def load_patch_labels(mask_path: str, keep_idx: np.ndarray,
                       patch_grid: int, patch_size: int) -> np.ndarray:
    """
    Load a segmentation mask and return the median class label for each
    16×16 pixel tile corresponding to the patch tokens at keep_idx.

    Returns: (n_keep,) int64 array
    """
    mask = np.load(mask_path)[MASK_KEY]                     # (4096, 4096)
    rows = keep_idx // patch_grid                            # (n_keep,)
    cols = keep_idx  % patch_grid

    r_offsets = np.arange(patch_size)
    c_offsets = np.arange(patch_size)
    # tile_rows/tile_cols: (n_keep, patch_size)
    tile_rows = rows[:, None] * patch_size + r_offsets[None, :]
    tile_cols = cols[:, None] * patch_size + c_offsets[None, :]
    # tiles: (n_keep, patch_size, patch_size)
    tiles = mask[tile_rows[:, :, None], tile_cols[:, None, :]]
    tiles_flat = tiles.reshape(len(keep_idx), -1)            # (n_keep, patch_size²)
    return np.round(np.median(tiles_flat, axis=1)).astype(np.int64)


def extract_patch_labels(mask_paths: list, keep_idx: np.ndarray,
                          patch_grid: int, patch_size: int) -> np.ndarray:
    """
    For each sample, load the corresponding mask and compute per-patch labels.

    Returns: (N_samples, n_keep) int64
    """
    all_labels = np.empty((len(mask_paths), len(keep_idx)), dtype=np.int64)
    for i, path in enumerate(tqdm(mask_paths, desc="Loading mask labels")):
        all_labels[i] = load_patch_labels(path, keep_idx, patch_grid, patch_size)
    return all_labels


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
    keep_idx: np.ndarray,
) -> np.ndarray:
    """
    Extract per-patch embeddings for spatial tokens at keep_idx positions.

    With nglo=0 the backbone returns (B, N_spatial, embed_dim) — all tokens
    are spatial patch tokens in raster order (row-major, top-left origin).

    Returns: (N_samples, n_keep, embed_dim) float32
    """
    model.eval()
    all_embs = []
    for batch in tqdm(dataloader, desc="Extracting patch embeddings"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        tokens = model(batch)                        # (B, N_spatial, D)
        kept   = tokens[:, keep_idx, :]              # (B, n_keep, D)
        all_embs.append(kept.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)          # (N_samples, n_keep, D)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def plot_umap_by_mask_class(emb2d: np.ndarray, flat_labels: np.ndarray,
                             out_path: Path, title_suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for cls_id, cls_name in sorted(MASK_CLASS_NAMES.items()):
        m = flat_labels == cls_id
        if not m.any():
            continue
        ax.scatter(emb2d[m, 0], emb2d[m, 1],
                   c=MASK_PALETTE[cls_id],
                   label=f"{cls_name}  (n={m.sum():,})",
                   s=4, alpha=0.5, linewidths=0)
    ax.legend(title="Mask class", loc="best", markerscale=3)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Surya patch embeddings – UMAP by mask class{title_suffix}")
    _save(fig, out_path)


def plot_umap_by_position(emb2d: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                          patch_grid: int, out_path: Path,
                          title_suffix: str = "") -> None:
    hue  = cols / patch_grid
    val  = 1.0 - rows / patch_grid
    sat  = np.ones_like(hue) * 0.8
    rgba = mcolors.hsv_to_rgb(np.stack([hue, sat, val], axis=1))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(emb2d[:, 0], emb2d[:, 1], c=rgba, s=4, alpha=0.5, linewidths=0)
    sm_h = plt.cm.ScalarMappable(cmap="hsv",    norm=mcolors.Normalize(0, patch_grid))
    sm_v = plt.cm.ScalarMappable(cmap="gray_r", norm=mcolors.Normalize(0, patch_grid))
    sm_h.set_array([]); sm_v.set_array([])
    plt.colorbar(sm_h, ax=ax, fraction=0.03, pad=0.01).set_label("patch column →")
    plt.colorbar(sm_v, ax=ax, fraction=0.03, pad=0.06).set_label("patch row ↓")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"Surya patch embeddings – UMAP by spatial position{title_suffix}")
    _save(fig, out_path)


def plot_spatial_magnitude(patch_embs: np.ndarray, keep_idx: np.ndarray,
                            patch_grid: int, out_path: Path,
                            title_suffix: str = "") -> None:
    """Heatmap of mean embedding L2 norm at each sampled patch position."""
    norms      = np.linalg.norm(patch_embs, axis=-1)   # (N_samples, n_keep)
    mean_norms = norms.mean(axis=0)                     # (n_keep,)

    rows = keep_idx // patch_grid
    cols = keep_idx  % patch_grid

    grid = np.full((patch_grid, patch_grid), np.nan)
    for i, (r, c) in enumerate(zip(rows, cols)):
        grid[r, c] = mean_norms[i]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, origin="upper", cmap="inferno",
                   vmin=np.nanpercentile(grid, 2),
                   vmax=np.nanpercentile(grid, 98))
    plt.colorbar(im, ax=ax).set_label("Embedding L2 norm")
    ax.set_xlabel("Patch column"); ax.set_ylabel("Patch row")
    ax.set_title(f"Spatial map of embedding magnitude{title_suffix}")
    _save(fig, out_path)


def plot_spatial_mask_labels(patch_labels: np.ndarray, keep_idx: np.ndarray,
                              patch_grid: int, out_path: Path,
                              title_suffix: str = "") -> None:
    """
    Spatial heatmap of the mask class at each sampled patch position for the last sample.

    Unsampled positions are shown in light gray.
    """
    # (n_keep,) — label per position for the last sample
    median_labels = patch_labels[-1].astype(np.int64)

    rows = keep_idx // patch_grid
    cols = keep_idx  % patch_grid

    # Build a float grid initialised to NaN (unsampled positions stay gray)
    grid = np.full((patch_grid, patch_grid), np.nan)
    for i, (r, c) in enumerate(zip(rows, cols)):
        grid[r, c] = float(median_labels[i])

    # Discrete colour map from MASK_PALETTE: classes -1, 0, 1, 2
    classes  = sorted(MASK_CLASS_NAMES.keys())          # [-1, 0, 1, 2]
    palette  = [MASK_PALETTE[k] for k in classes]
    cmap     = mcolors.ListedColormap(palette)
    bounds   = [c - 0.5 for c in classes] + [classes[-1] + 0.5]
    norm     = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 6))
    # Background for unsampled patches
    ax.set_facecolor("#eeeeee")
    im = ax.imshow(grid, origin="upper", cmap=cmap, norm=norm,
                   interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, ticks=classes)
    cbar.ax.set_yticklabels([MASK_CLASS_NAMES[c] for c in classes])
    ax.set_xlabel("Patch column")
    ax.set_ylabel("Patch row")
    ax.set_title(f"Spatial map of mask label (last sample){title_suffix}")
    _save(fig, out_path)


def plot_silhouette(flat_embs: np.ndarray, flat_labels: np.ndarray,
                    out_path: Path, metric: str = "cosine") -> None:
    """Patch-level silhouette score for each mask class (excludes NA / off-disk patches)."""
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.preprocessing import LabelEncoder

    # Exclude off-disk patches (class -1) from the silhouette calculation
    valid = flat_labels != -1
    embs, lbls = flat_embs[valid], flat_labels[valid]

    if len(np.unique(lbls)) < 2:
        print("  silhouette: need ≥2 classes — skipping")
        return

    le          = LabelEncoder()
    lbls_int    = le.fit_transform(lbls)
    score       = silhouette_score(embs, lbls_int, metric=metric)
    samp_scores = silhouette_samples(embs, lbls_int, metric=metric)
    cls_names   = [MASK_CLASS_NAMES.get(c, str(c)) for c in le.classes_]
    cls_means   = [samp_scores[lbls_int == i].mean() for i in range(len(le.classes_))]
    print(f"  silhouette (metric, {len(le.classes_)} classes): {score:.4f}")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(cls_names, cls_means,
           color=[MASK_PALETTE.get(c, "steelblue") for c in le.classes_])
    ax.axhline(score, color="red", linestyle="--", label=f"Overall: {score:.3f}")
    ax.set_xlabel("Mask class"); ax.set_ylabel("Mean silhouette")
    ax.set_title("Silhouette score by mask class\n(cosine distance, patch-level embeddings)")
    ax.legend()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# HDBSCAN clustering
# ---------------------------------------------------------------------------
def cluster_embeddings(emb2d: np.ndarray, min_cluster_size: int,
                        min_samples: int | None = None,
                        cluster_selection_method = "eom",
                        cluster_selection_epsilon = 0.5,
                        alpha = 1.0) -> np.ndarray:
    """Run HDBSCAN on the full-dimensional embedding space.  Noise points receive label -1."""
    try:
        from cuml.cluster import hdbscan
    except ImportError:
        raise ImportError("hdbscan is required: conda install -c conda-forge hdbscan")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size = min_cluster_size,
        min_samples      = min_samples,
        metric           = "euclidean",
        cluster_selection_method = cluster_selection_method,
        cluster_selection_epsilon = cluster_selection_epsilon,
        alpha = alpha

    )
    labels     = clusterer.fit_predict(emb2d)
    n_clusters = int((labels >= 0).any() and labels.max() + 1)
    n_noise    = int((labels == -1).sum())
    print(f"  HDBSCAN: {n_clusters} clusters, "
          f"{n_noise:,} noise points ({100 * n_noise / len(labels):.1f}%)")
    return labels


def _cluster_colormap(unique_cluster_ids: list[int]) -> dict[int, np.ndarray]:
    """Map integer cluster IDs → RGBA colours.  Noise (-1) is always light gray."""
    real_ids = [c for c in unique_cluster_ids if c >= 0]
    colors   = plt.cm.tab20(np.linspace(0, 1, max(len(real_ids), 1)))
    cmap     = {cid: colors[i] for i, cid in enumerate(real_ids)}
    cmap[-1] = np.array([0.75, 0.75, 0.75, 1.0])
    return cmap


def plot_umap_by_cluster(emb2d: np.ndarray, cluster_labels: np.ndarray,
                          out_path: Path, title_suffix: str = "") -> None:
    """UMAP scatter coloured by HDBSCAN cluster label (noise in light gray)."""
    unique_ids = sorted(set(cluster_labels.tolist()))
    cmap       = _cluster_colormap(unique_ids)

    fig, ax = plt.subplots(figsize=(8, 7))
    for cid in unique_ids:
        m     = cluster_labels == cid
        label = "Noise" if cid == -1 else f"Cluster {cid}  (n={m.sum():,})"
        ax.scatter(emb2d[m, 0], emb2d[m, 1],
                   c=[cmap[cid]], label=label,
                   s=4, alpha=0.5, linewidths=0)
    n_real = sum(1 for c in unique_ids if c >= 0)
    if n_real <= 20:
        ax.legend(title="HDBSCAN cluster", loc="best",
                  markerscale=3, fontsize=7, ncol=max(1, n_real // 10))
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"Surya patch embeddings – HDBSCAN clusters{title_suffix}")
    _save(fig, out_path)


def plot_spatial_cluster_labels(cluster_labels: np.ndarray, n_samples: int,
                                  keep_idx: np.ndarray, patch_grid: int,
                                  out_path: Path, title_suffix: str = "") -> None:
    """
    Spatial heatmap of the HDBSCAN cluster at each patch position for the last sample.

    cluster_labels is the flat (N*n_keep,) array from cluster_embeddings().
    Noise (-1) and unsampled positions are shown in light gray.
    """
    import matplotlib.patches as mpatches

    n_keep    = len(keep_idx)
    dominant  = cluster_labels.reshape(n_samples, n_keep)[-1].astype(int)

    rows = keep_idx // patch_grid
    cols = keep_idx  % patch_grid

    unique_ids = sorted(set(dominant.tolist()))
    cmap       = _cluster_colormap(unique_ids)

    # Build RGBA image; background colour for unsampled patches
    bg    = np.array([0.93, 0.93, 0.93, 1.0])
    img   = np.tile(bg, (patch_grid, patch_grid, 1))
    for i, (r, c) in enumerate(zip(rows, cols)):
        img[r, c] = cmap[int(dominant[i])]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(img, origin="upper", interpolation="nearest")

    handles = []
    if -1 in unique_ids:
        handles.append(mpatches.Patch(color=cmap[-1], label="Noise"))
    for cid in (c for c in unique_ids if c >= 0):
        handles.append(mpatches.Patch(color=cmap[cid], label=f"Cluster {cid}"))
    if len(handles) <= 20:
        ax.legend(handles=handles, loc="best",
                  fontsize=7, ncol=max(1, len(handles) // 10))

    ax.set_xlabel("Patch column"); ax.set_ylabel("Patch row")
    ax.set_title(f"Spatial map of HDBSCAN cluster (last sample){title_suffix}")
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

    print(f"  Intersecting with masks in {args.masks_dir} …")
    mask_df  = load_mask_timestamps(args.masks_dir)
    print(f"  {len(mask_df)} mask files found")
    index_df = intersect_with_masks(index_df, mask_df, MASKS_TIME_TOLERANCE)
    print(f"  {len(index_df):,} samples after mask intersection")

    embed_df = subsample_index(index_df, args.start_date, args.end_date, args.max_samples)
    embed_index_path = args.output_dir / "embedding_index.csv"
    embed_df.drop(columns=["mask_path"]).to_csv(embed_index_path, index=False)
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
        s3_storage_options       = {"anon": True},
        s3_download_to_temp      = True,
        s3_cache_dir             = CACHE_PATH,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=DEVICE.type == "cuda",
    )
    timestamps = pd.to_datetime([str(ts) for ts in dataset.valid_indices])
    print(f"  {len(dataset)} samples  "
          f"({timestamps[0].date()} → {timestamps[-1].date()})")

    # Align mask_paths to the timestamps that the dataset actually loaded.
    # The dataset may drop samples it cannot read, so we look up by timestamp
    # rather than assuming positional correspondence with embed_df.
    ts_to_mask = dict(zip(embed_df["timestep"], embed_df["mask_path"]))
    mask_paths = [ts_to_mask[ts] for ts in timestamps]

    # ── 4. Patch indices (shared by embedding extraction and label extraction)
    n_spatial = PATCH_GRID * PATCH_GRID
    keep_idx  = np.sort(rng.choice(n_spatial, size=min(args.max_patches, n_spatial), replace=False))

    rows = (keep_idx // PATCH_GRID).astype(np.int32)
    cols = (keep_idx  % PATCH_GRID).astype(np.int32)

    # ── 4. Extract patch embeddings ───────────────────────────────────────
    print(f"\n[4/6] Extracting patch embeddings "
          f"({len(keep_idx)} patches/sample) …")
    patch_embs = extract_patch_embeddings(backbone, loader, DEVICE, keep_idx)
    # (N_samples, n_keep, embed_dim)

    patch_pos = np.broadcast_to(
        np.stack([rows, cols], axis=1)[None],
        (patch_embs.shape[0], len(keep_idx), 2),
    ).copy()

    if args.normalize_embeddings:
        norms = np.linalg.norm(patch_embs, axis=-1, keepdims=True)
        patch_embs = patch_embs / np.where(norms == 0, 1.0, norms)
        print("  Embeddings L2-normalised to unit magnitude")

    np.save(args.output_dir / "patch_embeddings.npy", patch_embs)
    np.save(args.output_dir / "patch_positions.npy",  patch_pos)
    np.save(args.output_dir / "timestamps.npy",        timestamps.values)
    print(f"  patch_embs shape: {patch_embs.shape}")

    # ── 5. Mask labels ────────────────────────────────────────────────────
    print("\n[5/6] Extracting mask labels …")
    patch_labels = extract_patch_labels(mask_paths, keep_idx, PATCH_GRID, PATCH_SIZE)
    # (N_samples, n_keep)
    np.save(args.output_dir / "patch_mask_labels.npy", patch_labels)

    N, n_keep, D = patch_embs.shape
    flat_embs   = patch_embs.reshape(-1, D)          # (N*n_keep, D)
    flat_labels = patch_labels.reshape(-1)            # (N*n_keep,)
    flat_rows   = patch_pos[:, :, 0].reshape(-1)
    flat_cols   = patch_pos[:, :, 1].reshape(-1)

    for cls_id, cls_name in sorted(MASK_CLASS_NAMES.items()):
        print(f"    {cls_name}: {(flat_labels == cls_id).sum():,} patches")

    # ── 6. UMAP + HDBSCAN + figures ──────────────────────────────────────
    print("\n[6/6] Running UMAP, HDBSCAN, and saving figures …")
    try:
        from cuml import manifold as umap_lib
    except ImportError:
        raise ImportError("cuML is required: conda install -c rapidsai cuml")

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

    print("  Running HDBSCAN …")
    cluster_labels = cluster_embeddings(
        flat_embs,
        min_cluster_size = args.hdbscan_min_cluster_size,
        min_samples      = args.hdbscan_min_samples,
    )
    np.save(args.output_dir / "patch_cluster_labels.npy", cluster_labels)

    title_suffix = (
        f"\n{args.start_date} → {args.end_date}  "
        f"({N} samples × {n_keep} patches = {len(flat_embs):,} points)"
    )

    plot_umap_by_mask_class(
        emb2d, flat_labels,
        args.output_dir / "umap_by_mask_class.png",
        title_suffix,
    )
    plot_umap_by_cluster(
        emb2d, cluster_labels,
        args.output_dir / "umap_by_cluster.png",
        title_suffix,
    )
    plot_umap_by_position(
        emb2d, flat_rows, flat_cols, PATCH_GRID,
        args.output_dir / "umap_by_position.png",
        title_suffix,
    )
    plot_spatial_magnitude(
        patch_embs, keep_idx, PATCH_GRID,
        args.output_dir / "spatial_emb_magnitude.png",
        f"\n({N} samples)",
    )
    plot_spatial_mask_labels(
        patch_labels, keep_idx, PATCH_GRID,
        args.output_dir / "spatial_mask_labels.png",
        f"\n({N} samples)",
    )
    plot_spatial_cluster_labels(
        cluster_labels, N, keep_idx, PATCH_GRID,
        args.output_dir / "spatial_cluster_labels.png",
        f"\n({N} samples)",
    )
    plot_silhouette(
        flat_embs, flat_labels,
        args.output_dir / "silhouette.png",
        metric="euclidean",
    )

    print(f"\nDone.  All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
