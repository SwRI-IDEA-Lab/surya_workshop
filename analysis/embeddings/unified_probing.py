#!/usr/bin/env python3
"""
unified_probing.py — Quantitative + visual probing of Surya patch embeddings.

This script consumes the outputs of ``embedding_ablation.py`` and applies a
suite of probes to Surya's per-patch token embeddings.  Every probe runs
twice — once on raw embeddings, once on position-mean-ablated residuals —
so each result has a position-vs-content attribution.

Probes
------
1.  effective_rank      exp(entropy of normalized singular values).  Overall
                        and stratified by SPoCA mask class.
2.  pca_ev              Cumulative PCA explained-variance curves, overall and
                        stratified by mask class.  The class-stratified
                        version asks how many dimensions the model spends on
                        AR vs QS vs CH structure.
3.  linear_probe_px     Ridge LOO/k-fold CV predicting per-channel patch-
                        averaged pixel values from token embeddings.  Tests
                        information preservation.  Reported per channel and
                        per channel group.
4.  linear_probe_cls    Multinomial logistic LOO/k-fold CV predicting SPoCA
                        mask class from token embeddings.  Reports accuracy
                        plus full confusion matrix — model and labeler may
                        disagree in informative ways.
5.  spatial_corr        Pearson correlation between pairwise embedding
                        distance and pairwise pixel-grid distance.  The
                        raw-vs-residual delta is the share of token spatial
                        autocorrelation driven by position encoding.
6.  umap_cluster        2D UMAP on residuals + clustering (K-means K=4 by
                        default to match SPoCA's class count).  Lifts the
                        machinery from embedding_ablation_umap.py.
7.  cluster_purity      Cluster vs SPoCA mask class confusion matrix and
                        purity-weighted accuracy.  A cluster that's 70% AR
                        and 30% QS along the AR boundary is plausibly the
                        model picking up plage, which SPoCA doesn't have a
                        class for.
8.  pca_viz             Per-sample 3-component PCA RGB spatial map.  Optional
                        AnyUp super-resolution (--use-anyup; off by default).

Inputs
------
``--ablation-dir`` : the output directory of ``embedding_ablation.py``.
Required files in that directory:
    patch_residuals.npy             (N, n_keep, D)        float32
    patch_pos_mean.npy              (N_spatial, D)        float32
    patch_residuals_positions.npy   (n_keep, 2)           int32  [row, col]
    patch_mask_labels.npy           (N, n_keep)           int64
    timestamps.npy                  (N,)
    extraction_index.csv            CSV with 'timestep' column

Optional (used if present, regenerated otherwise):
    patch_residuals_pca.npy         (N, n_keep, k_pca)    float32

Outputs
-------
<out-dir>/
    effective_rank/        Probe 1 results
    pca_ev/                Probe 2 results
    linear_probe_px/       Probe 3 results
    linear_probe_cls/      Probe 4 results
    spatial_corr/          Probe 5 results
    umap_cluster/          Probe 6 results
    cluster_purity/        Probe 7 results
    pca_viz/               Probe 8 results
    unified_report.md      Top-level summary across probes

Channel groups for the linear probe targets
-------------------------------------------
We can't decompose Surya's token embeddings by modality (channels are
flattened in the patch projection from layer 1).  The grouping below
applies only to the *targets* of the linear probe — the per-group probe
score reports how well the embedding linearly predicts the average
patch-pixel value within that input channel group.

    AIA-coronal-hot   {94, 131}
    AIA-corona        {171, 193, 211, 335}
    AIA-chromosphere  {304, 1600}
    HMI-vector        {B_x, B_y, B_z}
    HMI-LOS           {B_los}
    HMI-velocity      {V_los}

Assumptions baked in (questioning recommended)
----------------------------------------------
* raw[t,i,:] = residuals[t,i,:] + mu[keep_idx[i],:] is an exact
  reconstruction (up to float precision).  Verified at runtime by sanity
  check; toggle off via --reextract-raw to re-run the model forward and
  cross-check.
* Mask class -1 (NA = off-disk) is excluded from class-stratified probes
  by default.  A separate "with-NA" set is kept for the 4-class probe so
  the ability of the embedding to identify off-disk geometry is also
  reported.
* The dataset's batch dict contains exactly one tensor of shape
  (B, 13, T, 4096, 4096); auto-detected by shape rather than by key
  (because HelioNetCDFDataset's exact key naming isn't visible from the
  embedding_ablation.py imports).
* For the linear-probe-against-mask-class result: this jointly tests the
  model AND SPoCA's thresholds.  A confusion matrix is always reported so
  that systematic disagreements (e.g., Surya cluster boundary at plage
  that SPoCA collapses into AR) are visible rather than averaged away.
* CV scheme: LOO when N <= 20, 5-fold when N > 20.  Override with
  --cv-loo or --cv-folds.

Compute envelope
----------------
Single-GPU friendly.  Targeted at <30 minutes for a 200-sample × 8192-
patch ablation run, excluding the optional --reextract-raw pass.  cuML is
required for UMAP and K-means; sklearn handles Ridge, PCA, and effective-
rank SVD.
"""

# ============================================================================
# IMPORTS & SETUP
# ============================================================================

import argparse
import gc
import json
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from tqdm.auto import tqdm

# --- Repo root resolution: same convention as the upstream scripts ---------
_script_dir = Path(__file__).resolve().parent
_repo_root  = _script_dir
while not (_repo_root / "workshop_infrastructure").exists() and _repo_root != _repo_root.parent:
    _repo_root = _repo_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Reuse upstream constants and helpers.  This is the canonical way to make
# the unified script stay in lockstep with embedding_ablation.py — if the
# upstream changes (e.g., new channel ordering), this script picks it up.
from embedding_analysis import (
    INDEX_PATH,
    SCALERS_PATH,
    CHECKPOINT_PATH,
    CACHE_PATH,
    CHANNELS,
    MODEL_CONFIG,
    PATCH_SIZE,
    PATCH_GRID,
    DEVICE,
    MASK_CLASS_NAMES,
    MASK_PALETTE,
    build_backbone,
)
from workshop_infrastructure.datasets.helio import HelioNetCDFDataset
from workshop_infrastructure.utils import build_scalers


# ============================================================================
# CHANNEL GROUPS
# ============================================================================
# Indexed against the CHANNELS constant from embedding_analysis.py:
#   0:aia94 1:aia131 2:aia171 3:aia193 4:aia211
#   5:aia304 6:aia335 7:aia1600
#   8:hmi_m  9:hmi_bx 10:hmi_by 11:hmi_bz 12:hmi_v
#
# Re-derived from CHANNELS at module load to survive any reordering upstream.
def _idx(name: str) -> int:
    return CHANNELS.index(name)

CHANNEL_GROUPS = {
    "AIA-coronal-hot":  [_idx("aia94"),  _idx("aia131")],
    "AIA-corona":       [_idx("aia171"), _idx("aia193"), _idx("aia211"), _idx("aia335")],
    "AIA-chromosphere": [_idx("aia304"), _idx("aia1600")],
    "HMI-vector":       [_idx("hmi_bx"), _idx("hmi_by"), _idx("hmi_bz")],
    "HMI-LOS":          [_idx("hmi_m")],   # B_los — split off per the user's request
    "HMI-velocity":     [_idx("hmi_v")],
}

GROUP_PALETTE = {
    "AIA-coronal-hot":  "#7b1fa2",
    "AIA-corona":       "#1976d2",
    "AIA-chromosphere": "#f57c00",
    "HMI-vector":       "#388e3c",
    "HMI-LOS":          "#5d4037",
    "HMI-velocity":     "#c2185b",
}


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ablation-dir", type=Path, required=True,
                   help="Output directory from embedding_ablation.py.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: <ablation-dir>/unified_probing/).")

    # Raw-embedding handling
    p.add_argument("--reextract-raw", action="store_true",
                   help="Re-run model forward to obtain raw embeddings (slow). "
                        "Default: reconstruct raw = residuals + mu[keep_idx], "
                        "which is exact up to float precision.")

    # Probe selection
    all_probes = ["effective_rank", "pca_ev", "linear_probe_px",
                  "linear_probe_px_stratified",
                  "linear_probe_cls", "spatial_corr", "umap_cluster",
                  "cluster_purity", "hidden_physical_spatial", "pca_viz"]
    p.add_argument("--probes", nargs="+", default=all_probes,
                   choices=all_probes,
                   help="Which probes to run.  Default: all.")

    # Cross-validation
    p.add_argument("--cv-folds", type=int, default=5,
                   help="K for k-fold CV when N > 20 (default 5).")
    p.add_argument("--cv-loo", action="store_true",
                   help="Force LOO-CV regardless of sample count.")

    # Linear probe knobs
    p.add_argument("--linear-probe-n-patches", type=int, default=1024,
                   help="Patches subsampled per sample for the linear probe.")
    p.add_argument("--ridge-alpha", type=float, default=1.0,
                   help="Ridge regularization (default 1.0).  Ignored when "
                        "--ridge-alpha-sweep is given.")
    p.add_argument("--ridge-alpha-sweep", nargs="?", const="auto", default=None,
                   help="Tune alpha by inner CV (RidgeCV / LogisticRegressionCV) "
                        "instead of using a fixed value.  Pass alone for the "
                        "default sweep [0.1, 1, 10, 100, 1000, 1e4], or pass a "
                        "comma-separated list (e.g. '1,10,100,1000').  For the "
                        "logistic probe, alphas are converted to Cs = 1/alpha.")
    p.add_argument("--use-pca-for-linear-probe", action="store_true",
                   help="Project both raw and residual embeddings onto the PCA "
                        "basis stored in <ablation-dir>/pca_components.npy "
                        "before the linear probes.  Strongly mitigates probe "
                        "overfitting on small-N runs (D=1280 -> k=50).  Both "
                        "conditions use the same residual-fit basis, so raw "
                        "vs residual stays apples-to-apples.")

    # PCA / effective-rank knobs
    p.add_argument("--pca-n-components", type=int, default=50,
                   help="Components fit for the PCA EV curves (default 50).")
    p.add_argument("--rank-subsample", type=int, default=50_000,
                   help="Rows subsampled for effective-rank SVD (default 50K).")

    # Spatial correlation knob
    p.add_argument("--spatial-corr-n-points", type=int, default=500,
                   help="Patches subsampled per sample for pairwise distance "
                        "calculation (default 500).  O(n^2) so keep modest.")

    # UMAP / clustering
    p.add_argument("--umap-n-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--umap-n-points", type=int, default=None,
                   help="Randomly subsample this many patches before UMAP "
                        "fit+transform.  Default: use all on-disk patches. "
                        "Strongly recommended for large runs: a 650-sample × "
                        "65536-patch ablation has ~20M on-disk patches, which "
                        "stresses even cuML UMAP.  500_000–1_000_000 gives "
                        "accurate 2D structure at manageable cost.")
    p.add_argument("--cluster-method", choices=["hdbscan", "kmeans", "gmm"],
                   default="hdbscan",
                   help="Clustering algorithm.  Default 'hdbscan' is the "
                        "right tool for token embeddings (variable cluster "
                        "shapes & sizes, data-driven K, explicit noise label "
                        "with Mahalanobis reassignment).  'kmeans' / 'gmm' "
                        "kept for cases where a fixed K is desired.")
    p.add_argument("--cluster-on", choices=["pca", "umap2d", "raw"],
                   default="pca",
                   help="Feature space for clustering.  Default 'pca' uses "
                        "the 50-PC residuals from <ablation-dir>/"
                        "patch_residuals_pca.npy, which is the preferred "
                        "regime: it preserves the dominant structure (~90%% "
                        "of residual variance), is high-dim enough for "
                        "Mahalanobis covariance estimation to be meaningful, "
                        "and avoids the well-known artifact that clusters "
                        "in 2D UMAP can be projection-induced rather than "
                        "real density modes.  'umap2d' clusters on the 2D "
                        "UMAP projection (cheap and visually appealing but "
                        "epistemically suspect: UMAP is non-distance-"
                        "preserving and density structure can be artifactual). "
                        "'raw' clusters on the full 1280-d residuals "
                        "(suffers high-dim distance concentration; usually "
                        "fails for HDBSCAN).")
    p.add_argument("--cluster-k", type=int, default=4,
                   help="K for K-means / GMM (ignored when "
                        "--cluster-method=hdbscan).")
    # HDBSCAN-specific knobs (ignored unless --cluster-method=hdbscan)
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=15,
                   help="HDBSCAN min_cluster_size (default 15).")
    p.add_argument("--hdbscan-min-samples", type=int, default=15,
                   help="HDBSCAN min_samples (default 15).")
    p.add_argument("--hdbscan-cluster-selection-method", default="leaf",
                   choices=["eom", "leaf"],
                   help="HDBSCAN selection method (default 'leaf').")
    p.add_argument("--hdbscan-cluster-selection-epsilon", type=float, default=2.0,
                   help="HDBSCAN cluster_selection_epsilon (default 2.0).")
    p.add_argument("--hdbscan-alpha", type=float, default=1.5,
                   help="HDBSCAN alpha (default 1.5).")
    p.add_argument("--mahalanobis-shrinkage", type=float, default=0.05,
                   help="Ledoit-Wolf-style shrinkage λ for cluster "
                        "covariances when reassigning HDBSCAN noise points: "
                        "Σ_reg = (1−λ)·Σ + λ·diag(Σ).  Default 0.05.")
    p.add_argument("--mahalanobis-reassign", action=argparse.BooleanOptionalAction,
                   default=True, dest="mahalanobis_reassign",
                   help="Reassign HDBSCAN noise points to their nearest "
                        "cluster via Mahalanobis distance (default: on).  "
                        "Use --no-mahalanobis-reassign to keep the -1 noise "
                        "label and skip reassignment.")
    p.add_argument("--exclude-na-from-clustering", action="store_true",
                   help="Filter out NA (off-disk) patches before UMAP+"
                        "clustering.  Highly recommended: NA patches form a "
                        "single dominant low-purity mega-cluster that washes "
                        "out the on-disk physical structure.  When this flag "
                        "is on, the saved cluster_labels.npy is full-length "
                        "with sentinel -2 marking excluded NA patches.")
    p.add_argument("--cluster-purity-top-n", type=int, default=30,
                   help="When K_hdbscan > this, the cluster-purity heatmap "
                        "shows only the top-N largest clusters (the rest "
                        "are summarized).  Default 30.")

    # PCA visualization
    p.add_argument("--pca-viz-n-samples", type=int, default=5,
                   help="Number of sample panels to render for the RGB PCA map.")
    p.add_argument("--pca-viz-on", choices=["raw", "residual", "both"],
                   default="both",
                   help="Which embedding space(s) to render PCA RGB for.  "
                        "'raw' = position+content; 'residual' = content-only "
                        "(after per-position mean ablation); 'both' shows "
                        "side-by-side panels per sample (default).")
    p.add_argument("--pca-viz-norm", choices=["per_sample", "global", "both"],
                   default="per_sample",
                   help="Color normalization for PCA RGB.  'per_sample' = "
                        "each sample's percentiles computed independently "
                        "(within-sample structure clear, cross-sample colors "
                        "incomparable in absolute terms; default).  'global' = "
                        "percentiles computed across all samples (cross-"
                        "sample comparisons valid; within-sample contrast "
                        "may be weaker).  'both' = render a 2×2 "
                        "raw/residual × per-sample/global comparison grid.")
    p.add_argument("--use-anyup", action="store_true",
                   help="Use AnyUp super-resolution for PCA RGB maps "
                        "(off by default; requires torch.hub network access).  "
                        "Applies to the residual variant only.")
    p.add_argument("--anyup-resolution", type=int, default=512)

    p.add_argument("--seed", type=int, default=42)

    # Probe-group shortcuts (mutually exclusive with each other; both
    # override --probes if given).
    _CLUSTERING_PROBES = ["umap_cluster", "cluster_purity", "hidden_physical_spatial"]
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--no-clustering", action="store_true",
                     help="Run all probes except umap_cluster, cluster_purity, "
                          "and hidden_physical_spatial.  Useful for a fast first "
                          "pass: PCA, linear probes, spatial correlation, and "
                          "PCA visualisation, without the expensive UMAP step.")
    grp.add_argument("--clustering-only", action="store_true",
                     help="Run only umap_cluster, cluster_purity, and "
                          "hidden_physical_spatial.  cluster_purity and "
                          "hidden_physical_spatial will load cluster_labels.npy "
                          "from a previous umap_cluster run if it exists.")

    args = p.parse_args()

    # Apply group shortcuts to args.probes so downstream code sees a plain list.
    if args.no_clustering:
        args.probes = [pr for pr in args.probes if pr not in _CLUSTERING_PROBES]
    elif args.clustering_only:
        args.probes = [pr for pr in _CLUSTERING_PROBES if pr in args.probes]

    return args


# ============================================================================
# DATA LOADING
# ============================================================================

@dataclass
class AblationOutputs:
    residuals:       np.ndarray   # (N, n_keep, D) float32
    mu:              np.ndarray   # (N_spatial, D) float32
    positions:       np.ndarray   # (n_keep, 2) int32 [row, col]
    keep_idx:        np.ndarray   # (n_keep,) int64 (raster index)
    mask_labels:     np.ndarray   # (N, n_keep) int64
    timestamps:      np.ndarray   # (N,)
    extraction_csv:  Path
    residuals_pca:   Optional[np.ndarray] = None  # (N, n_keep, k_pca)

    @property
    def N(self)        -> int: return self.residuals.shape[0]
    @property
    def n_keep(self)   -> int: return self.residuals.shape[1]
    @property
    def embed_dim(self)-> int: return self.residuals.shape[2]


def load_ablation_outputs(ablation_dir: Path) -> AblationOutputs:
    """Load all artifacts produced by embedding_ablation.py."""
    print(f"\n[load] Reading ablation outputs from {ablation_dir}")
    residuals   = np.load(ablation_dir / "patch_residuals.npy",   mmap_mode="r")
    mu          = np.load(ablation_dir / "patch_pos_mean.npy")
    positions   = np.load(ablation_dir / "patch_residuals_positions.npy")
    mask_labels = np.load(ablation_dir / "patch_mask_labels.npy", mmap_mode="r")
    timestamps  = np.load(ablation_dir / "timestamps.npy", allow_pickle=True)

    # Reconstruct keep_idx from positions (raster row * grid + col).
    keep_idx = (positions[:, 0].astype(np.int64) * PATCH_GRID
                + positions[:, 1].astype(np.int64))

    # Optional PCA file (used to speed up UMAP)
    pca_path = ablation_dir / "patch_residuals_pca.npy"
    pca = np.load(pca_path, mmap_mode="r") if pca_path.exists() else None

    print(f"       residuals    : {residuals.shape} {residuals.dtype}")
    print(f"       mu           : {mu.shape} {mu.dtype}")
    print(f"       mask_labels  : {mask_labels.shape}")
    print(f"       N={residuals.shape[0]}  n_keep={residuals.shape[1]}  "
          f"D={residuals.shape[2]}")
    if pca is not None:
        print(f"       pca residuals: {pca.shape} (loaded)")
    else:
        print(f"       pca residuals: not present (UMAP probe will compute on the fly)")

    return AblationOutputs(
        residuals=residuals.astype(np.float32, copy=False),
        mu=mu.astype(np.float32, copy=False),
        positions=positions.astype(np.int32, copy=False),
        keep_idx=keep_idx,
        mask_labels=mask_labels.astype(np.int64, copy=False),
        timestamps=timestamps,
        extraction_csv=ablation_dir / "extraction_index.csv",
        residuals_pca=pca.astype(np.float32, copy=False) if pca is not None else None,
    )


def reconstruct_raw_from_residuals(ab: AblationOutputs) -> np.ndarray:
    """raw[t,i,:] = residuals[t,i,:] + mu[keep_idx[i],:].  Exact within float."""
    print("\n[raw] Reconstructing raw embeddings from residuals + mu[keep_idx]")
    mu_kept = ab.mu[ab.keep_idx]                        # (n_keep, D)
    raw = ab.residuals + mu_kept[None, :, :]            # (N, n_keep, D), broadcast
    raw = raw.astype(np.float32, copy=False)
    print(f"      raw shape    : {raw.shape}")
    print(f"      raw mean/std : {raw.mean():.4f} / {raw.std():.4f}")
    print(f"      res mean/std : {ab.residuals.mean():.4f} / {ab.residuals.std():.4f}")
    return raw


def reextract_raw_embeddings(ab: AblationOutputs) -> np.ndarray:
    """Re-run model forward and gather raw embeddings at keep_idx positions."""
    print("\n[raw] Re-extracting raw embeddings via model forward (slow)")
    backbone = build_backbone(CHECKPOINT_PATH, DEVICE)
    scalers  = build_scalers(SCALERS_PATH)
    dataset = HelioNetCDFDataset(
        index_path                = str(ab.extraction_csv),
        time_delta_input_minutes  = [0],
        time_delta_target_minutes = 60,
        n_input_timestamps        = 1,
        rollout_steps             = 0,
        scalers                   = scalers,
        channels                  = CHANNELS,
        phase                     = "embed",
        load_forecast_frames      = False,
        s3_storage_options        = {"anon": True},
        s3_download_to_temp       = True,
        s3_cache_dir              = CACHE_PATH,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2,
        pin_memory=DEVICE.type == "cuda",
    )

    raw = np.empty((ab.N, ab.n_keep, ab.embed_dim), dtype=np.float32)
    backbone.eval()
    with torch.no_grad():
        for t, batch in enumerate(tqdm(loader, total=ab.N, desc="Re-extracting")):
            batch_d = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}
            tokens = backbone(batch_d)              # (1, N_spatial, D)
            raw[t] = tokens[0, ab.keep_idx].cpu().float().numpy()
    return raw


def _find_input_tensor(batch: dict) -> torch.Tensor:
    """Locate the (B, 13, T, 4096, 4096)-shaped tensor in a HelioNetCDFDataset
    batch dict.  Auto-detected by shape so we don't need to know the key name.
    """
    expected_C = len(CHANNELS)
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            continue
        # Match (..., C, T, H, W) where C == 13 and H == W == 4096.
        if v.ndim >= 4 and v.shape[-1] == 4096 and v.shape[-2] == 4096:
            # The channel dim is whichever non-batch axis equals 13.
            for ax, sz in enumerate(v.shape):
                if sz == expected_C:
                    return v
    raise RuntimeError(
        "Could not locate input tensor in batch dict. "
        f"Keys: {list(batch.keys())}; shapes: "
        f"{[(k, getattr(v, 'shape', type(v))) for k, v in batch.items()]}"
    )


def load_patch_pixel_targets(ab: AblationOutputs) -> np.ndarray:
    """Load patch-averaged input pixel values aligned to ablation keep_idx.

    Returns
    -------
    pixels : (N, n_keep, n_channels) float32
        Mean input value within each 16x16 patch tile, normalized by the
        same scaler the embedding was computed under (so the units match
        what the model actually saw).
    """
    print("\n[pixels] Loading patch-averaged input targets via dataset reload")
    scalers = build_scalers(SCALERS_PATH)
    dataset = HelioNetCDFDataset(
        index_path                = str(ab.extraction_csv),
        time_delta_input_minutes  = [0],
        time_delta_target_minutes = 60,
        n_input_timestamps        = 1,
        rollout_steps             = 0,
        scalers                   = scalers,
        channels                  = CHANNELS,
        phase                     = "embed",
        load_forecast_frames      = False,
        s3_storage_options        = {"anon": True},
        s3_download_to_temp       = True,
        s3_cache_dir              = CACHE_PATH,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2,
    )

    n_channels = len(CHANNELS)
    pixels = np.empty((ab.N, ab.n_keep, n_channels), dtype=np.float32)
    rows = ab.positions[:, 0].astype(np.int64)
    cols = ab.positions[:, 1].astype(np.int64)
    keep_idx = ab.keep_idx

    for t, batch in enumerate(tqdm(loader, total=ab.N, desc="Loading pixels")):
        x = _find_input_tensor(batch)                  # (1, C, T, H, W)
        x = x[0].cpu().numpy().astype(np.float32)      # (C, T, H, W)
        # Sum across time (T should be 1; if not, use mean).
        x = x.mean(axis=1)                             # (C, H, W)
        # Reshape into patches and average within each patch.
        # (C, PATCH_GRID, PATCH_SIZE, PATCH_GRID, PATCH_SIZE) -> mean over patch
        x_patch = (
            x.reshape(n_channels, PATCH_GRID, PATCH_SIZE, PATCH_GRID, PATCH_SIZE)
             .mean(axis=(2, 4))                        # (C, PATCH_GRID, PATCH_GRID)
             .reshape(n_channels, -1)                  # (C, N_spatial)
        )
        pixels[t] = x_patch[:, keep_idx].T             # (n_keep, C)

    print(f"        pixel targets: {pixels.shape}")
    return pixels


# ============================================================================
# UTILITIES
# ============================================================================

def _decide_cv(n_samples: int, args: argparse.Namespace) -> tuple[str, int]:
    """Returns (scheme_name, n_splits)."""
    if args.cv_loo or n_samples <= 20:
        return ("LOO", n_samples)
    return (f"{args.cv_folds}-fold", args.cv_folds)


def _subsample_rows(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if X.shape[0] <= n:
        return X
    idx = rng.choice(X.shape[0], n, replace=False)
    return X[idx]


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved -> {path}")


def _safe_json(obj):
    """Recursively make an object JSON-serializable."""
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _pca_reduce_for_probe(raw: np.ndarray, residuals: np.ndarray,
                           ab: AblationOutputs,
                           args: argparse.Namespace,
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Project raw and residual embeddings onto the residual PCA basis.

    The PCA was fit on flat residuals by ``embedding_ablation.py`` (sklearn
    PCA, which centers by default).  We recompute the centering mean from
    residuals here and apply the same mean to raw — this gives apples-to-
    apples 50-d inputs to the linear probe.  Ridge / LogisticRegression's
    bias term absorbs the (small) raw-vs-residual mean offset.

    Returns
    -------
    raw_proj, res_proj : (N, n_keep, k) each — k is the number of PCs.
    """
    components_path = args.ablation_dir / "pca_components.npy"
    if not components_path.exists():
        warnings.warn(
            f"--use-pca-for-linear-probe was set but {components_path} not "
            f"found; falling back to full-D embeddings."
        )
        return raw, residuals

    components = np.load(components_path).astype(np.float32)   # (k, D)
    k, D = components.shape
    if D != ab.embed_dim:
        warnings.warn(
            f"PCA components have D={D} but residuals have D={ab.embed_dim}; "
            f"falling back to full-D embeddings."
        )
        return raw, residuals

    flat_res = residuals.reshape(-1, D)
    pca_mean = flat_res.mean(axis=0).astype(np.float32)        # (D,)

    flat_raw = raw.reshape(-1, D)
    raw_proj = ((flat_raw - pca_mean) @ components.T).astype(np.float32)
    res_proj = ((flat_res - pca_mean) @ components.T).astype(np.float32)

    raw_proj = raw_proj.reshape(ab.N, ab.n_keep, k)
    res_proj = res_proj.reshape(ab.N, ab.n_keep, k)
    print(f"\n[pca-reduce] Linear-probe inputs: D={D} -> k={k}  "
          f"(residual-fit PCA basis applied to both conditions)")
    return raw_proj, res_proj


# ============================================================================
# HDBSCAN + Mahalanobis noise reassignment
# ============================================================================
# Adapted from embedding_ablation_umap.py.  K-means imposes spherical-cluster
# and equal-cluster-size assumptions that are violated by token embeddings;
# HDBSCAN gives data-driven K, variable cluster shapes, and an explicit noise
# label.  Mahalanobis reassignment turns the noise label into a covariance-
# aware hard assignment when needed for downstream comparisons.

def _build_cluster_mahalanobis_model(
    X: np.ndarray,
    labels: np.ndarray,
    lam: float,
) -> tuple:
    """Build per-cluster Cholesky covariance factors for Mahalanobis scoring.

    Parameters
    ----------
    X      : (N, F) float32 — feature matrix (only labelled points, no noise).
    labels : (N,) int32     — cluster IDs (must all be ≥ 0).
    lam    : float          — Ledoit-Wolf shrinkage λ for
                             Σ_reg = (1−λ)·Σ + λ·diag(Σ).

    Returns
    -------
    uniq         : (K,) int32  sorted cluster IDs present in labels.
    centroids    : (K, F) float32 cluster means.
    chol_factors : list of ("full", L) or ("diag", var_d) — one per cluster.
    log_dets     : (K,) float64 log|Σ_reg| per cluster.
    n_diag       : int  number of clusters that fell back to diagonal Σ.
    """
    from scipy.linalg import LinAlgError
    X = np.ascontiguousarray(X.astype(np.float32, copy=False))
    _N, F = X.shape
    lam = float(lam)
    uniq = np.array(sorted({int(k) for k in labels if k >= 0}), dtype=np.int32)
    K = len(uniq)
    centroids = np.stack([X[labels == k].mean(axis=0) for k in uniq])
    chol_factors: list = []
    log_dets = np.empty(K, dtype=np.float64)
    n_diag = 0
    for i, k in enumerate(uniq):
        members = X[labels == k]
        n_k = len(members)
        if n_k <= F:
            var_d = members.var(axis=0, ddof=1) + 1e-6
            chol_factors.append(("diag", var_d))
            log_dets[i] = float(np.log(var_d).sum())
            n_diag += 1
            continue
        cov = np.cov(members, rowvar=False).astype(np.float64)
        diag_cov = np.diag(np.diag(cov))
        cov_reg = (1.0 - lam) * cov + lam * diag_cov
        ridge = 1e-6 * np.trace(cov_reg) / F
        cov_reg.flat[:: F + 1] += ridge
        try:
            L = np.linalg.cholesky(cov_reg)
            chol_factors.append(("full", L))
            log_dets[i] = float(2.0 * np.log(np.diag(L)).sum())
        except LinAlgError:
            var_d = np.diag(cov_reg) + 1e-6
            chol_factors.append(("diag", var_d))
            log_dets[i] = float(np.log(var_d).sum())
            n_diag += 1
    return uniq, centroids, chol_factors, log_dets, n_diag


def _mahalanobis_assign_chunks(
    X_query: np.ndarray,
    uniq: np.ndarray,
    centroids: np.ndarray,
    chol_factors: list,
    log_dets: np.ndarray,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Assign each row of X_query to the nearest cluster by Mahalanobis distance.

    Score = Mahalanobis²(x, k) + log|Σ_k| (Gaussian log-density surrogate;
    lower = closer).  Chunked to keep peak memory bounded.

    Parameters
    ----------
    X_query      : (M, F) — points to assign.
    uniq / centroids / chol_factors / log_dets — from _build_cluster_mahalanobis_model.
    chunk_size   : rows processed per iteration.

    Returns
    -------
    assigned : (M,) int32 cluster IDs drawn from uniq.
    """
    from scipy.linalg import solve_triangular
    X_query = np.ascontiguousarray(X_query.astype(np.float32, copy=False))
    M = len(X_query)
    K = len(uniq)
    assigned = np.empty(M, dtype=np.int32)
    for start in range(0, M, chunk_size):
        chunk = X_query[start:start + chunk_size]
        scores = np.empty((len(chunk), K), dtype=np.float64)
        for i in range(K):
            delta = (chunk - centroids[i]).astype(np.float64)
            kind, factor = chol_factors[i]
            if kind == "full":
                y = solve_triangular(factor, delta.T, lower=True,
                                     check_finite=False)
                d2 = (y * y).sum(axis=0)
            else:   # diag
                d2 = ((delta * delta) / factor).sum(axis=1)
            scores[:, i] = d2 + log_dets[i]
        assigned[start:start + chunk_size] = uniq[scores.argmin(axis=1)]
    return assigned


def _hdbscan_with_mahalanobis_reassignment(
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str = "leaf",
    cluster_selection_epsilon: float = 2.0,
    alpha: float = 1.5,
    mahalanobis_shrinkage: float = 0.05,
    reassign_noise: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run HDBSCAN, then (optionally) reassign noise points to nearest
    cluster in Mahalanobis distance with Ledoit-Wolf shrinkage.

    Parameters
    ----------
    X : (N, F) — feature matrix in the clustering space.
    min_cluster_size, min_samples, cluster_selection_method,
        cluster_selection_epsilon, alpha : HDBSCAN knobs.
    mahalanobis_shrinkage : λ for Σ_reg = (1−λ)·Σ + λ·diag(Σ).
        Stabilizes near-singular cluster covariances.
    reassign_noise : if False, leave noise labels (-1) in place.

    Returns
    -------
    labels_with_noise : (N,) int — original HDBSCAN labels (-1 = noise).
    labels_assigned   : (N,) int — every point assigned to a real cluster
        (identical to labels_with_noise if reassign_noise is False).
    info : dict with keys n_clusters, n_noise, noise_frac, n_diag_fallback.
    """
    try:
        from cuml.cluster import hdbscan
    except ImportError:
        raise ImportError("cuML required for HDBSCAN; install via "
                          "`conda install -c rapidsai cuml`")

    X = np.ascontiguousarray(X.astype(np.float32, copy=False))
    N, F = X.shape

    print(f"   HDBSCAN: min_cluster_size={min_cluster_size}, "
          f"min_samples={min_samples}, selection={cluster_selection_method}, "
          f"epsilon={cluster_selection_epsilon}, alpha={alpha}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size = min_cluster_size,
        min_samples      = min_samples,
        metric           = "euclidean",
        cluster_selection_method = cluster_selection_method,
        cluster_selection_epsilon = cluster_selection_epsilon,
        alpha            = alpha,
    )
    labels_with_noise = clusterer.fit_predict(X).astype(np.int32)
    K_total = int(labels_with_noise.max()) + 1 if (labels_with_noise >= 0).any() else 0
    n_noise = int((labels_with_noise == -1).sum())
    noise_frac = n_noise / max(N, 1)
    print(f"   HDBSCAN: {K_total} clusters, {n_noise:,} noise points "
          f"({100*noise_frac:.1f}%)")

    if K_total == 0:
        warnings.warn("HDBSCAN found no clusters at the given thresholds; "
                      "all points labeled noise.")
        return labels_with_noise, labels_with_noise.copy(), {
            "n_clusters": 0, "n_noise": n_noise,
            "noise_frac": noise_frac, "n_diag_fallback": 0,
        }

    if not reassign_noise or n_noise == 0:
        return labels_with_noise, labels_with_noise.copy(), {
            "n_clusters": K_total, "n_noise": n_noise,
            "noise_frac": noise_frac, "n_diag_fallback": 0,
        }

    # --- Mahalanobis-aware reassignment of noise points ---------------------
    # Build model on the labelled (non-noise) subset; reuses the two helpers
    # so the same distance metric applies here and in the label-extension step.
    lam = float(mahalanobis_shrinkage)
    non_noise_mask = labels_with_noise >= 0
    uniq, centroids, chol_factors, log_dets, n_diag = _build_cluster_mahalanobis_model(
        X[non_noise_mask], labels_with_noise[non_noise_mask], lam,
    )
    K_real = len(uniq)

    if n_diag:
        print(f"   {n_diag}/{K_real} clusters used diagonal-Σ fallback "
              f"(small or rank-deficient).")

    noise_mask = labels_with_noise == -1
    noise_pts  = X[noise_mask]
    print(f"   reassigning {len(noise_pts):,} noise points (mahalanobis, λ={lam}) ...")
    assigned = _mahalanobis_assign_chunks(
        noise_pts, uniq, centroids, chol_factors, log_dets,
    )

    labels_assigned = labels_with_noise.copy()
    labels_assigned[noise_mask] = assigned

    info = {"n_clusters": K_real, "n_noise": n_noise,
            "noise_frac": noise_frac, "n_diag_fallback": n_diag}
    return labels_with_noise, labels_assigned, info


def _cluster_palette(K: int) -> np.ndarray:
    """Return (K, 4) RGBA palette that scales gracefully past 20 clusters."""
    if K <= 0:
        return np.empty((0, 4))
    if K <= 20:
        return plt.cm.tab20(np.linspace(0, 1, K))
    # For K > 20, switch to a perceptually-uniform cyclic colormap.
    # Adjacent clusters get distinct hues; legend won't be displayed anyway.
    hues = np.linspace(0, 1, K, endpoint=False)
    sat = np.full(K, 0.7)
    val = np.full(K, 0.95)
    rgb = mcolors.hsv_to_rgb(np.stack([hues, sat, val], axis=1))
    return np.concatenate([rgb, np.ones((K, 1))], axis=1)


# ============================================================================
# PROBE 1 — EFFECTIVE RANK
# ============================================================================
def _effective_rank(X: np.ndarray) -> float:
    """exp(entropy of normalized singular values)."""
    if X.shape[0] < 2 or X.shape[1] < 2:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    s = s[s > 1e-10]
    if s.size == 0:
        return float("nan")
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def probe_effective_rank(raw: np.ndarray, residuals: np.ndarray,
                          mask_labels: np.ndarray, args: argparse.Namespace,
                          out_dir: Path) -> dict:
    """Effective rank, overall and stratified by SPoCA class.

    Stratified ranks are clipped to a common subsample size so the numbers
    are comparable across classes (rank scales with sample count).
    """
    print("\n[probe 1/8] Effective rank")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    flat_lbl = mask_labels.reshape(-1)
    flat_raw = raw.reshape(-1, raw.shape[-1])
    flat_res = residuals.reshape(-1, residuals.shape[-1])

    results = {}
    cap = args.rank_subsample
    classes = [(None, "overall", flat_raw, flat_res, flat_lbl)]
    for cls_id in (-1, 0, 1, 2):
        if cls_id == -1:   # NA: keep separately, mostly for sanity
            tag = "NA"
        else:
            tag = MASK_CLASS_NAMES[cls_id]
        m = flat_lbl == cls_id
        if m.sum() < 100:
            print(f"   {tag:14s}: only {m.sum()} samples, skipping")
            continue
        # Subsample indices before extracting from the memmap to avoid
        # materialising hundreds of GiB for large classes (e.g. NA ~32M patches).
        idx = np.where(m)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
            idx.sort()
        classes.append((cls_id, tag, flat_raw[idx], flat_res[idx], flat_lbl[idx]))

    for _cls_id, tag, R, Rr, _lbl in classes:
        R_sub  = _subsample_rows(R,  cap, rng)
        Rr_sub = _subsample_rows(Rr, cap, rng)
        rk_raw = _effective_rank(R_sub)
        rk_res = _effective_rank(Rr_sub)
        delta  = rk_raw - rk_res
        D      = R.shape[1]
        results[tag] = {
            "raw":   rk_raw,
            "residual": rk_res,
            "delta_position":  delta,
            "embed_dim": D,
            "n_subsample": min(cap, R.shape[0]),
        }
        print(f"   {tag:14s}: raw={rk_raw:7.2f}  residual={rk_res:7.2f}  "
              f"Δ(position)={delta:+6.2f}   (D={D})")

    # Bar chart
    tags = list(results.keys())
    raw_vals = [results[t]["raw"]      for t in tags]
    res_vals = [results[t]["residual"] for t in tags]
    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(tags)), 4))
    ax.bar(x - 0.2, raw_vals, 0.4, label="Raw embeddings",      color="#aaa")
    ax.bar(x + 0.2, res_vals, 0.4, label="Residual (post-ablation)", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=20, ha="right")
    ax.set_ylabel("Effective rank")
    ax.axhline(raw.shape[-1], ls="--", color="gray", alpha=0.5,
               label=f"D = {raw.shape[-1]}")
    ax.set_title("Effective rank — raw vs position-ablated residuals")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir / "effective_rank.png")

    (out_dir / "effective_rank.json").write_text(
        json.dumps(_safe_json(results), indent=2)
    )
    return results


# ============================================================================
# PROBE 2 — PCA EXPLAINED-VARIANCE CURVES
# ============================================================================
def probe_pca_ev(raw: np.ndarray, residuals: np.ndarray,
                  mask_labels: np.ndarray, args: argparse.Namespace,
                  out_dir: Path) -> dict:
    """Cumulative PCA explained-variance, overall and stratified by mask class.

    Per-class curves are fit on patches with that label only, on equal
    subsample sizes for fairness.
    """
    print("\n[probe 2/8] PCA explained-variance curves")
    from sklearn.decomposition import PCA
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    flat_lbl = mask_labels.reshape(-1)
    flat_raw = raw.reshape(-1, raw.shape[-1])
    flat_res = residuals.reshape(-1, residuals.shape[-1])
    cap = args.rank_subsample
    nc = args.pca_n_components

    def _ev(X: np.ndarray) -> np.ndarray:
        if X.shape[0] < nc + 1:
            return np.zeros(nc, dtype=np.float32)
        Xs = _subsample_rows(X, cap, rng)
        return PCA(n_components=nc, random_state=args.seed,
                   svd_solver="randomized").fit(Xs).explained_variance_ratio_

    # Overall + per-class (excluding NA)
    panels = [("overall", flat_raw, flat_res)]
    for cls_id in (0, 1, 2):
        tag = MASK_CLASS_NAMES[cls_id]
        m = flat_lbl == cls_id
        if m.sum() < nc + 100:
            continue
        # Subsample indices first to avoid materialising large memmap slices.
        idx = np.where(m)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
            idx.sort()
        panels.append((tag, flat_raw[idx], flat_res[idx]))

    results = {}
    for tag, R, Rr in panels:
        ev_raw = _ev(R)
        ev_res = _ev(Rr)
        results[tag] = {
            "explained_variance_raw":      ev_raw.tolist(),
            "explained_variance_residual": ev_res.tolist(),
            "cumulative_raw":      np.cumsum(ev_raw).tolist(),
            "cumulative_residual": np.cumsum(ev_res).tolist(),
            "n_subsample": min(cap, R.shape[0]),
        }
        # Where does each curve cross common thresholds?
        for thresh in (0.5, 0.9, 0.95):
            cum = np.cumsum(ev_raw)
            results[tag][f"raw_n_to_{int(thresh*100)}pct"] = (
                int(np.searchsorted(cum, thresh)) + 1 if cum[-1] >= thresh else None
            )
            cum = np.cumsum(ev_res)
            results[tag][f"residual_n_to_{int(thresh*100)}pct"] = (
                int(np.searchsorted(cum, thresh)) + 1 if cum[-1] >= thresh else None
            )

    # Plot 1: overall — raw vs residual
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(1, nc + 1)
    ax.plot(xs, results["overall"]["cumulative_raw"],      "-o", ms=3,
            label="Raw embeddings",                    color="#888")
    ax.plot(xs, results["overall"]["cumulative_residual"], "-s", ms=3,
            label="Residual (post-ablation)",          color="#1f77b4")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA EV — overall (raw vs residual)")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    _save_fig(fig, out_dir / "pca_ev_overall.png")

    # Plot 2: per-class (residual only — the meaningful comparison)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cls_id in (0, 1, 2):
        tag = MASK_CLASS_NAMES[cls_id]
        if tag not in results:
            continue
        ax.plot(xs, results[tag]["cumulative_residual"], "-",
                color=MASK_PALETTE[cls_id], label=tag)
    ax.plot(xs, results["overall"]["cumulative_residual"], "k--",
            alpha=0.6, label="Overall")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance (residual)")
    ax.set_title("PCA EV by SPoCA class (residual embeddings)")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    _save_fig(fig, out_dir / "pca_ev_by_class.png")

    (out_dir / "pca_ev.json").write_text(
        json.dumps(_safe_json(results), indent=2)
    )
    return results


# ============================================================================
# PROBE 3 — LINEAR PROBE: EMBEDDINGS -> PIXEL CHANNELS
# ============================================================================
def probe_linear_probe_pixels(raw: np.ndarray, residuals: np.ndarray,
                                pixels: np.ndarray, args: argparse.Namespace,
                                out_dir: Path) -> dict:
    """Per-channel Ridge LOO/k-fold CV.  Reports mean R² across folds.

    With ``--ridge-alpha-sweep``, alpha is tuned on each training fold by
    sklearn's RidgeCV (efficient leave-one-out generalized CV).  The chosen
    alphas are reported per (channel, condition) so you can see whether
    raw and residual probes prefer different regularization strengths —
    big differences suggest the two representations have very different
    effective dimensionalities.
    """
    print("\n[probe 3/8] Linear probe — embeddings -> per-channel pixels")
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.metrics import r2_score
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Resolve alpha schedule
    sweep = args.ridge_alpha_sweep
    if sweep is None:
        alphas = [args.ridge_alpha]
        sweep_msg = f"alpha={alphas[0]} (fixed)"
    elif sweep == "auto":
        alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
        sweep_msg = f"alphas={alphas} (inner-CV)"
    else:
        alphas = [float(x) for x in sweep.split(",")]
        sweep_msg = f"alphas={alphas} (inner-CV)"
    use_cv = len(alphas) > 1

    N, n_keep, D = raw.shape
    n_ch = pixels.shape[-1]
    scheme, k = _decide_cv(N, args)
    print(f"   CV scheme  : {scheme} ({k} splits)  N={N}")
    print(f"   Probe input: D={D} "
          f"({'PCA-reduced' if args.use_pca_for_linear_probe else 'full'})")
    print(f"   Ridge      : {sweep_msg}")

    # Common subsample of patch positions for all channels.
    sub = min(args.linear_probe_n_patches, n_keep)
    sub_idx = rng.choice(n_keep, sub, replace=False)
    print(f"   Subsampling {sub} of {n_keep} patches per sample")

    def _kfold_indices(n: int, k: int, scheme: str):
        """Yield (train_idx, test_idx) tuples."""
        if scheme == "LOO":
            for i in range(n):
                yield np.array([j for j in range(n) if j != i]), np.array([i])
        else:
            order = rng.permutation(n)
            folds = np.array_split(order, k)
            for i in range(k):
                test = folds[i]
                train = np.concatenate([folds[j] for j in range(k) if j != i])
                yield train, test

    results = {"raw": {}, "residual": {}, "scheme": scheme,
               "k_splits": k, "n_patches_subsample": sub,
               "alphas_swept": alphas, "alpha_sweep_active": use_cv,
               "use_pca_for_probe": bool(args.use_pca_for_linear_probe),
               "embed_dim": D}

    for cond_name, X_full in [("raw", raw), ("residual", residuals)]:
        X_sub = X_full[:, sub_idx, :]                 # (N, sub, D)
        y_sub = pixels[:, sub_idx, :]                 # (N, sub, C)

        for c in range(n_ch):
            y_c = y_sub[..., c]                       # (N, sub)
            r2_per_fold = []
            chosen_alphas = []
            for tr, te in _kfold_indices(N, k, scheme):
                X_tr = X_sub[tr].reshape(-1, D)
                y_tr = y_c[tr].ravel()
                X_te = X_sub[te].reshape(-1, D)
                y_te = y_c[te].ravel()
                if use_cv:
                    mdl = RidgeCV(alphas=alphas)
                else:
                    mdl = Ridge(alpha=alphas[0])
                mdl.fit(X_tr, y_tr)
                r2_per_fold.append(r2_score(y_te, mdl.predict(X_te)))
                chosen_alphas.append(
                    float(mdl.alpha_) if use_cv else float(alphas[0])
                )
            entry = {
                "r2_mean": float(np.mean(r2_per_fold)),
                "r2_std":  float(np.std(r2_per_fold)),
            }
            if use_cv:
                entry["alpha_median"]   = float(np.median(chosen_alphas))
                entry["alpha_per_fold"] = chosen_alphas
            results[cond_name][CHANNELS[c]] = entry

        # Group means (mean of channel R² within each group)
        results[cond_name]["_group_r2"] = {}
        for gname, idxs in CHANNEL_GROUPS.items():
            r2s = [results[cond_name][CHANNELS[i]]["r2_mean"] for i in idxs]
            results[cond_name]["_group_r2"][gname] = float(np.mean(r2s))

        print(f"   {cond_name:9s}: " + " ".join(
            f"{g}={results[cond_name]['_group_r2'][g]:+.3f}"
            for g in CHANNEL_GROUPS
        ))
        if use_cv:
            for gname, idxs in CHANNEL_GROUPS.items():
                ams = [results[cond_name][CHANNELS[i]]["alpha_median"] for i in idxs]
                print(f"     {gname:18s}  median α = {np.median(ams):g}")

    # Plot per-channel R² (raw vs residual) grouped by channel group
    n_groups = len(CHANNEL_GROUPS)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, cond in zip(axes, ["raw", "residual"]):
        xs = []
        labels = []
        colors = []
        bars = []
        x_cursor = 0
        for gname, idxs in CHANNEL_GROUPS.items():
            for i in idxs:
                xs.append(x_cursor)
                labels.append(CHANNELS[i])
                colors.append(GROUP_PALETTE[gname])
                bars.append(results[cond][CHANNELS[i]]["r2_mean"])
                x_cursor += 1
            x_cursor += 0.6   # gap between groups
        ax.bar(xs, bars, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(r"R$^2$")
        title = f"Linear probe pixel R² — {cond}"
        if use_cv:
            title += "  (inner-CV α)"
        if args.use_pca_for_linear_probe:
            title += "  [PCA-reduced inputs]"
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    # Legend
    handles = [plt.Rectangle((0,0), 1, 1, color=GROUP_PALETTE[g]) for g in CHANNEL_GROUPS]
    axes[0].legend(handles, list(CHANNEL_GROUPS.keys()),
                   loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    _save_fig(fig, out_dir / "linear_probe_px.png")

    (out_dir / "linear_probe_px.json").write_text(
        json.dumps(_safe_json(results), indent=2)
    )
    return results


# ============================================================================
# PROBE 3b — LINEAR PROBE: STRATIFIED BY SPOCA CLASS
# ============================================================================
def _kfold_indices_over_n(n: int, k: int, scheme: str,
                            rng: np.random.Generator):
    """Yield (train_subset_idx, test_subset_idx) over n items.

    Used by the stratified probe where the number of valid samples per
    SPoCA class can be < total N (a sample with zero AR patches is invalid
    for the AR stratum), so the CV scheme must adapt to whatever n is.
    """
    if scheme == "LOO":
        for i in range(n):
            yield np.array([j for j in range(n) if j != i]), np.array([i])
    else:
        order = rng.permutation(n)
        folds = np.array_split(order, k)
        for i in range(k):
            test = folds[i]
            train = np.concatenate([folds[j] for j in range(k) if j != i])
            yield train, test


def probe_linear_probe_pixels_stratified(raw: np.ndarray, residuals: np.ndarray,
                                           pixels: np.ndarray,
                                           mask_labels: np.ndarray,
                                           args: argparse.Namespace,
                                           out_dir: Path) -> dict:
    """Per-class-stratified pixel linear probe.

    Same Ridge LOO/k-fold machinery as ``probe_linear_probe_pixels``, but
    restricts patches to a single SPoCA class at a time (CH, QS, AR; NA
    excluded since by construction there's no on-disk pixel to predict).

    Tests whether the position-vs-content attribution found at the global
    level is uniform across physical region types.  Asymmetries here are
    physically informative — e.g., HMI-LOS may carry more content
    information at AR locations than at QS locations because that's where
    the magnetogram signal is strongest above the noise floor.

    Implementation notes
    --------------------
    * Patches are subsampled per (sample, class) up to
      ``--linear-probe-n-patches`` patches.  Samples with zero patches of a
      given class are skipped for that class only, so each class can use a
      different effective N.
    * CV scheme adapts to the per-class effective N: LOO when N_class <= 20,
      else k-fold.  This keeps the probe meaningful even for rarely-active
      classes on quiet sun days.
    """
    print("\n[probe 3b] Linear probe — stratified by SPoCA class")
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.metrics import r2_score
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Resolve alpha schedule (mirrors the global pixel probe)
    sweep = args.ridge_alpha_sweep
    if sweep is None:
        alphas = [args.ridge_alpha]
    elif sweep == "auto":
        alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    else:
        alphas = [float(x) for x in sweep.split(",")]
    use_cv = len(alphas) > 1

    N, n_keep, D = raw.shape
    n_ch = pixels.shape[-1]
    n_per_cap = args.linear_probe_n_patches

    classes = [(0, "Coronal Hole"), (1, "Quiet Sun"), (2, "Active Region")]

    results = {
        "alphas_swept": alphas, "alpha_sweep_active": use_cv,
        "use_pca_for_probe": bool(args.use_pca_for_linear_probe),
        "embed_dim": D,
        "n_per_class_per_sample_cap": n_per_cap,
        "by_class": {},
    }

    for cls_id, cls_name in classes:
        # Per-sample patch indices for this class
        per_sample_idx: dict[int, np.ndarray] = {}
        for t in range(N):
            cls_patches = np.where(mask_labels[t] == cls_id)[0]
            if len(cls_patches) == 0:
                continue
            n_take = min(n_per_cap, len(cls_patches))
            sub = rng.choice(cls_patches, n_take, replace=False)
            per_sample_idx[t] = sub

        valid_t = sorted(per_sample_idx.keys())
        n_valid = len(valid_t)
        if n_valid < 3:
            print(f"   {cls_name}: only {n_valid} valid samples, skipping")
            continue

        avg_n = float(np.mean([len(per_sample_idx[t]) for t in valid_t]))
        # Adapt CV scheme to per-class effective N
        if args.cv_loo or n_valid <= 20:
            local_scheme, local_k = "LOO", n_valid
        else:
            local_scheme, local_k = f"{args.cv_folds}-fold", args.cv_folds
        print(f"   {cls_name:14s}: {n_valid}/{N} samples valid, "
              f"avg {avg_n:.0f} patches/sample, CV={local_scheme}")

        results["by_class"][cls_name] = {
            "raw": {}, "residual": {},
            "n_valid_samples": n_valid,
            "avg_patches_per_sample": avg_n,
            "scheme": local_scheme, "k_splits": local_k,
        }

        for cond_name, X_full in [("raw", raw), ("residual", residuals)]:
            for c in range(n_ch):
                X_per = [X_full[t, per_sample_idx[t], :] for t in valid_t]
                y_per = [pixels[t, per_sample_idx[t], c] for t in valid_t]

                r2_per_fold = []
                chosen_alphas = []
                for tr_idx, te_idx in _kfold_indices_over_n(
                        n_valid, local_k, local_scheme, rng):
                    X_tr = np.concatenate([X_per[i] for i in tr_idx])
                    y_tr = np.concatenate([y_per[i] for i in tr_idx])
                    X_te = np.concatenate([X_per[i] for i in te_idx])
                    y_te = np.concatenate([y_per[i] for i in te_idx])
                    if use_cv:
                        mdl = RidgeCV(alphas=alphas)
                    else:
                        mdl = Ridge(alpha=alphas[0])
                    mdl.fit(X_tr, y_tr)
                    r2_per_fold.append(r2_score(y_te, mdl.predict(X_te)))
                    chosen_alphas.append(
                        float(mdl.alpha_) if use_cv else float(alphas[0])
                    )
                entry = {
                    "r2_mean": float(np.mean(r2_per_fold)),
                    "r2_std":  float(np.std(r2_per_fold)),
                }
                if use_cv:
                    entry["alpha_median"] = float(np.median(chosen_alphas))
                results["by_class"][cls_name][cond_name][CHANNELS[c]] = entry

            # Per-group means
            results["by_class"][cls_name][cond_name]["_group_r2"] = {
                gname: float(np.mean([
                    results["by_class"][cls_name][cond_name][CHANNELS[i]]["r2_mean"]
                    for i in idxs
                ]))
                for gname, idxs in CHANNEL_GROUPS.items()
            }
            print(f"     {cond_name:9s}: " + " ".join(
                f"{g}={results['by_class'][cls_name][cond_name]['_group_r2'][g]:+.3f}"
                for g in CHANNEL_GROUPS
            ))

    # ------ Plotting ------------------------------------------------------
    # Heatmap: rows = channel groups, cols = SPoCA classes, panels = raw/residual
    cls_names = [name for _, name in classes
                  if name in results["by_class"]]
    if not cls_names:
        print("   no classes had enough samples; skipping plot")
        (out_dir / "linear_probe_px_stratified.json").write_text(
            json.dumps(_safe_json(results), indent=2)
        )
        return results

    group_names = list(CHANNEL_GROUPS.keys())
    n_groups = len(group_names)
    n_classes = len(cls_names)

    fig, axes = plt.subplots(1, 2, figsize=(2.6 * n_classes + 6, 4.2),
                               sharey=True)
    for ax, cond in zip(axes, ("raw", "residual")):
        mat = np.zeros((n_groups, n_classes))
        for j, cls_name in enumerate(cls_names):
            for i, gname in enumerate(group_names):
                mat[i, j] = results["by_class"][cls_name][cond]["_group_r2"][gname]
        # Symmetric colormap centered at 0 so negative R² shows clearly
        vmax = max(1.0, float(np.abs(mat).max()))
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(cls_names, rotation=20, ha="right")
        ax.set_yticks(range(n_groups))
        ax.set_yticklabels(group_names)
        ax.set_title(f"{cond}")
        for i in range(n_groups):
            for j in range(n_classes):
                ax.text(j, i, f"{mat[i,j]:+.2f}", ha="center", va="center",
                        color="white" if abs(mat[i,j]) > vmax*0.55 else "black",
                        fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("Linear probe pixel R² — stratified by SPoCA class\n"
                  "(channel-group means; rows = channel group, "
                  "cols = SPoCA class)")
    fig.tight_layout()
    _save_fig(fig, out_dir / "linear_probe_px_stratified.png")

    # Also: a "delta" heatmap (raw - residual) summarizing how much
    # position contributes per (group, class).
    fig, ax = plt.subplots(figsize=(2.6 * n_classes + 3, 4.2))
    delta = np.zeros((n_groups, n_classes))
    for j, cls_name in enumerate(cls_names):
        for i, gname in enumerate(group_names):
            r = results["by_class"][cls_name]["raw"]["_group_r2"][gname]
            x = results["by_class"][cls_name]["residual"]["_group_r2"][gname]
            delta[i, j] = r - x
    vmax = max(1.0, float(np.abs(delta).max()))
    im = ax.imshow(delta, aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(cls_names, rotation=20, ha="right")
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(group_names)
    for i in range(n_groups):
        for j in range(n_classes):
            ax.text(j, i, f"{delta[i,j]:+.2f}", ha="center", va="center",
                    color="white" if abs(delta[i,j]) > vmax*0.55 else "black",
                    fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title("Position-vs-content attribution per (group, class)\n"
                  "Δ = R²(raw) − R²(residual);  larger = more position-driven")
    fig.tight_layout()
    _save_fig(fig, out_dir / "linear_probe_px_stratified_delta.png")

    (out_dir / "linear_probe_px_stratified.json").write_text(
        json.dumps(_safe_json(results), indent=2)
    )
    return results


# ============================================================================
# PROBE 4 — LINEAR PROBE: EMBEDDINGS -> SPOCA CLASS
# ============================================================================
def probe_linear_probe_class(raw: np.ndarray, residuals: np.ndarray,
                              mask_labels: np.ndarray, args: argparse.Namespace,
                              out_dir: Path) -> dict:
    """Multinomial logistic LOO/k-fold CV; reports accuracy + confusion matrix.

    Two probe variants:
      - 4-class (incl. NA): tests off-disk separability too
      - 3-class (CH/QS/AR): tests on-disk physical class separability

    With ``--ridge-alpha-sweep``, the inverse-regularization C is tuned on
    each training fold by sklearn's LogisticRegressionCV (Cs = 1/alphas).
    """
    print("\n[probe 4/8] Linear probe — embeddings -> SPoCA class")
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.metrics import accuracy_score, confusion_matrix
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Resolve regularization schedule (mirrors the pixel-probe logic)
    sweep = args.ridge_alpha_sweep
    if sweep is None:
        alphas = [args.ridge_alpha]
        sweep_msg = f"C={1.0/alphas[0]:g} (fixed)"
    elif sweep == "auto":
        alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
        sweep_msg = f"alphas={alphas} -> Cs (inner-CV)"
    else:
        alphas = [float(x) for x in sweep.split(",")]
        sweep_msg = f"alphas={alphas} -> Cs (inner-CV)"
    use_cv = len(alphas) > 1
    Cs = [1.0 / a for a in alphas]

    N, n_keep, D = raw.shape
    scheme, k = _decide_cv(N, args)
    print(f"   CV scheme  : {scheme} ({k} splits)")
    print(f"   Probe input: D={D} "
          f"({'PCA-reduced' if args.use_pca_for_linear_probe else 'full'})")
    print(f"   Logistic   : {sweep_msg}")

    sub = min(args.linear_probe_n_patches, n_keep)
    sub_idx = rng.choice(n_keep, sub, replace=False)

    def _kfold_indices(n: int, k_: int, scheme_: str):
        if scheme_ == "LOO":
            for i in range(n):
                yield np.array([j for j in range(n) if j != i]), np.array([i])
        else:
            order = rng.permutation(n)
            folds = np.array_split(order, k_)
            for i in range(k_):
                test = folds[i]
                train = np.concatenate([folds[j] for j in range(k_) if j != i])
                yield train, test

    results = {"raw": {}, "residual": {}, "scheme": scheme, "k_splits": k,
               "alphas_swept": alphas, "alpha_sweep_active": use_cv,
               "use_pca_for_probe": bool(args.use_pca_for_linear_probe),
               "embed_dim": D}

    for cond_name, X_full in [("raw", raw), ("residual", residuals)]:
        X_sub = X_full[:, sub_idx, :]                 # (N, sub, D)
        y_sub = mask_labels[:, sub_idx]               # (N, sub)

        for variant_name, valid_classes in [
            ("4class", [-1, 0, 1, 2]),
            ("3class", [0, 1, 2]),
        ]:
            y_pred_all, y_true_all, chosen_Cs = [], [], []
            for tr, te in _kfold_indices(N, k, scheme):
                X_tr = X_sub[tr].reshape(-1, D)
                y_tr = y_sub[tr].ravel()
                X_te = X_sub[te].reshape(-1, D)
                y_te = y_sub[te].ravel()

                # Restrict to valid classes
                m_tr = np.isin(y_tr, valid_classes)
                m_te = np.isin(y_te, valid_classes)
                if m_tr.sum() < 10 or m_te.sum() < 1:
                    continue
                X_tr_v, y_tr_v = X_tr[m_tr], y_tr[m_tr]
                X_te_v, y_te_v = X_te[m_te], y_te[m_te]

                if len(np.unique(y_tr_v)) < 2:
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if use_cv:
                        # Inner CV folds: keep modest because each fold's
                        # training set is already small.  cv must be ≥2 and
                        # ≤ min class count for stratification to work.
                        min_class_count = int(np.min(np.bincount(
                            np.searchsorted(np.unique(y_tr_v), y_tr_v)
                        )))
                        inner_cv = max(2, min(3, min_class_count))
                        mdl = LogisticRegressionCV(
                            Cs=Cs, max_iter=200, solver="lbfgs",
                            n_jobs=1, cv=inner_cv,
                        )
                    else:
                        # multi_class was removed in sklearn 1.7; lbfgs
                        # auto-detects multinomial from the target.
                        mdl = LogisticRegression(
                            max_iter=200, solver="lbfgs", n_jobs=1,
                        )
                    mdl.fit(X_tr_v, y_tr_v)

                y_pred_all.append(mdl.predict(X_te_v))
                y_true_all.append(y_te_v)
                if use_cv:
                    # mdl.C_ is per-class for LogisticRegressionCV; take median
                    chosen_Cs.append(float(np.median(mdl.C_)))

            if not y_pred_all:
                continue
            y_pred = np.concatenate(y_pred_all)
            y_true = np.concatenate(y_true_all)
            cm = confusion_matrix(y_true, y_pred, labels=valid_classes)
            acc = accuracy_score(y_true, y_pred)
            entry = {
                "accuracy": float(acc),
                "confusion_matrix": cm.tolist(),
                "labels": valid_classes,
                "n_test": int(len(y_true)),
            }
            if use_cv and chosen_Cs:
                entry["C_median"] = float(np.median(chosen_Cs))
                entry["alpha_implied_median"] = float(1.0 / np.median(chosen_Cs))
            results[cond_name][variant_name] = entry

            msg = f"   {cond_name:9s} {variant_name}: accuracy={acc:.4f}  n={len(y_true):,}"
            if use_cv and chosen_Cs:
                msg += f"  median α = {1.0/np.median(chosen_Cs):g}"
            print(msg)

    # Confusion matrix plot — 3class for raw vs residual
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for row, cond in enumerate(["raw", "residual"]):
        for col, variant in enumerate(["4class", "3class"]):
            ax = axes[row, col]
            d = results[cond].get(variant)
            if d is None:
                ax.set_visible(False); continue
            cm = np.array(d["confusion_matrix"], dtype=float)
            cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
            lbls = [MASK_CLASS_NAMES[c] for c in d["labels"]]
            ax.set_xticks(range(len(lbls)))
            ax.set_yticks(range(len(lbls)))
            ax.set_xticklabels(lbls, rotation=30, ha="right")
            ax.set_yticklabels(lbls)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"{cond}  {variant}\nacc={d['accuracy']:.3f}")
            for i in range(len(lbls)):
                for j in range(len(lbls)):
                    txt = f"{cm_norm[i,j]:.2f}"
                    ax.text(j, i, txt, ha="center", va="center",
                            color="white" if cm_norm[i,j] > 0.5 else "black",
                            fontsize=8)
    fig.suptitle("Linear probe — SPoCA class confusion matrices "
                  "(row-normalized)")
    fig.tight_layout()
    _save_fig(fig, out_dir / "linear_probe_cls_confusion.png")

    (out_dir / "linear_probe_cls.json").write_text(
        json.dumps(_safe_json(results), indent=2)
    )
    return results


# ============================================================================
# PROBE 5 — SPATIAL CONSISTENCY CORRELATION
# ============================================================================
def probe_spatial_corr(raw: np.ndarray, residuals: np.ndarray,
                        positions: np.ndarray, args: argparse.Namespace,
                        out_dir: Path) -> dict:
    """Pearson correlation between pairwise embedding distance and pairwise
    pixel-grid distance, per sample, raw vs residual."""
    print("\n[probe 5/8] Spatial consistency correlation")
    from sklearn.metrics import pairwise_distances
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    N, n_keep, _ = raw.shape
    n_pts = min(args.spatial_corr_n_points, n_keep)
    print(f"   {n_pts} points × {N} samples (pairs computed per sample)")

    results = {"raw": [], "residual": []}
    for t in range(N):
        idx = rng.choice(n_keep, n_pts, replace=False)
        pos = positions[idx].astype(float)
        sd_dist = pairwise_distances(pos, metric="euclidean")
        tri = np.triu_indices(n_pts, k=1)
        spatial = sd_dist[tri]

        for cond_name, X_full in [("raw", raw), ("residual", residuals)]:
            ed = pairwise_distances(X_full[t, idx], metric="cosine")
            corr = float(np.corrcoef(ed[tri], spatial)[0, 1])
            results[cond_name].append(corr)

    summary = {
        "raw_mean":      float(np.mean(results["raw"])),
        "raw_std":       float(np.std(results["raw"])),
        "residual_mean": float(np.mean(results["residual"])),
        "residual_std":  float(np.std(results["residual"])),
        "delta_mean":    float(np.mean(results["raw"]) - np.mean(results["residual"])),
        "per_sample_raw":      results["raw"],
        "per_sample_residual": results["residual"],
        "n_points":      n_pts,
    }
    print(f"   raw      Pearson(dist_emb, dist_grid) = "
          f"{summary['raw_mean']:.4f} ± {summary['raw_std']:.4f}")
    print(f"   residual Pearson(dist_emb, dist_grid) = "
          f"{summary['residual_mean']:.4f} ± {summary['residual_std']:.4f}")
    print(f"   Δ(position-driven spatial correlation) = "
          f"{summary['delta_mean']:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([results["raw"], results["residual"]],
                tick_labels=["Raw", "Residual"], showmeans=True,
                meanline=True)
    ax.set_ylabel("Pearson(emb dist, grid dist)")
    ax.set_title(f"Spatial consistency — {N} samples × {n_pts} pts/sample")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save_fig(fig, out_dir / "spatial_corr.png")

    (out_dir / "spatial_corr.json").write_text(
        json.dumps(_safe_json(summary), indent=2)
    )
    return summary


# ============================================================================
# PROBE 6 — UMAP + CLUSTERING ON RESIDUALS
# ============================================================================
def probe_umap_cluster(ab: AblationOutputs, args: argparse.Namespace,
                        out_dir: Path) -> dict:
    """UMAP (always for visualization) + clustering (in --cluster-on space).

    Returns the cluster_labels array (consumed by probe_cluster_purity).

    Two distinct feature spaces are in play:
      • UMAP fit  : always uses the highest-dim residual representation
                    available (PCA-50 if present, else full 1280-d).  This
                    is purely for the 2D projection used in scatter plots.
      • Clustering: controlled by --cluster-on.  Default is 'pca' (50-d),
                    which is where HDBSCAN's density estimation is most
                    meaningful.  'umap2d' clusters on the 2D output of
                    UMAP — cheap but the density structure there is partly
                    a UMAP optimization artifact rather than a property of
                    the embedding itself.  'raw' clusters on the full
                    1280-d residuals — usually fails for HDBSCAN due to
                    high-dimensional distance concentration.
    """
    print("\n[probe 6/8] UMAP + clustering on residuals")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build the candidate feature spaces ---------------------------------
    raw_flat = ab.residuals.reshape(-1, ab.embed_dim)
    if ab.residuals_pca is not None:
        pca_flat = ab.residuals_pca.reshape(-1, ab.residuals_pca.shape[-1])
    else:
        pca_flat = None
    full_flat_lbl = ab.mask_labels.reshape(-1)
    n_total = len(full_flat_lbl)

    # --- Optional NA filter ------------------------------------------------
    # When --exclude-na-from-clustering is set, drop off-disk (-1) patches
    # before UMAP+clustering.  This dramatically clarifies on-disk physical
    # structure since NA otherwise forms a dominant low-purity mega-cluster.
    # The saved cluster_labels.npy is still full-length so downstream probes
    # can index it with the original mask_labels: -2 marks excluded patches.
    if args.exclude_na_from_clustering:
        keep_mask = full_flat_lbl != -1
        n_kept    = int(keep_mask.sum())
        n_dropped = n_total - n_kept
        print(f"   --exclude-na-from-clustering: dropping {n_dropped:,} NA "
              f"patches ({100*n_dropped/n_total:.1f}%); clustering on "
              f"{n_kept:,} on-disk patches")
    else:
        keep_mask = np.ones(n_total, dtype=bool)
        n_kept    = n_total

    raw_flat_clu = raw_flat[keep_mask]
    pca_flat_clu = pca_flat[keep_mask] if pca_flat is not None else None
    # Keep references to the full kept arrays before optional subsampling so
    # that cluster labels can later be extended to all n_kept patches.
    raw_flat_clu_full = raw_flat_clu
    pca_flat_clu_full = pca_flat_clu

    # --- Optional point-count cap for UMAP -----------------------------------
    # Without this, a 650-sample × 65536-patch run has ~20M on-disk patches —
    # intractable even for GPU-accelerated cuML UMAP.  Subsampling to 500K–1M
    # preserves the global density structure at manageable cost.
    rng_umap = np.random.default_rng(args.seed)
    umap_sel = None
    if args.umap_n_points is not None and n_kept > args.umap_n_points:
        umap_sel = rng_umap.choice(n_kept, size=args.umap_n_points, replace=False)
        umap_sel.sort()  # sorted access is friendlier to cache/mmap
        raw_flat_clu = raw_flat_clu[umap_sel]
        if pca_flat_clu is not None:
            pca_flat_clu = pca_flat_clu[umap_sel]
        print(f"   --umap-n-points: subsampled {args.umap_n_points:,} of "
              f"{n_kept:,} on-disk patches for UMAP")

    # Build a mask covering exactly the points that entered UMAP+clustering.
    # When subsampling was applied, this is a strict subset of keep_mask.
    if umap_sel is not None:
        keep_idx = np.where(keep_mask)[0]
        umap_keep_mask = np.zeros(n_total, dtype=bool)
        umap_keep_mask[keep_idx[umap_sel]] = True
    else:
        umap_keep_mask = keep_mask

    # UMAP fit always uses the highest-D representation available — purely
    # for the 2D projection used in scatter plots.
    if pca_flat_clu is not None:
        umap_input = pca_flat_clu
        umap_space_tag = f"PCA residuals ({pca_flat_clu.shape[1]}-d)"
    else:
        umap_input = raw_flat_clu
        umap_space_tag = f"raw residuals ({ab.embed_dim}-d)"
    print(f"   UMAP input space: {umap_space_tag}")

    # cuML UMAP
    print(f"   Running UMAP (n_neighbors={args.umap_n_neighbors}, "
          f"min_dist={args.umap_min_dist})")
    try:
        from cuml import manifold as umap_lib
    except ImportError:
        raise ImportError("cuML required for the umap_cluster probe; "
                          "install via: conda install -c rapidsai cuml")
    reducer = umap_lib.UMAP(
        n_neighbors  = args.umap_n_neighbors,
        min_dist     = args.umap_min_dist,
        n_components = 2,
        random_state = args.seed,
        verbose      = False,
    )
    emb2d = reducer.fit_transform(umap_input)
    np.save(out_dir / "umap_2d.npy", emb2d)

    # --- Resolve the clustering feature space (--cluster-on) ---------------
    # All candidate spaces use the *filtered* arrays so that NA-exclusion
    # propagates to clustering regardless of --cluster-on choice.
    if args.cluster_on == "pca":
        if pca_flat_clu is None:
            warnings.warn(
                "--cluster-on pca requested but no patch_residuals_pca.npy "
                "found in --ablation-dir; falling back to raw residuals.  "
                "Re-run embedding_ablation.py with --pca-components > 0 to "
                "generate the PCA file."
            )
            cluster_input = raw_flat_clu
            cluster_space_tag = f"raw residuals ({raw_flat_clu.shape[1]}-d)"
        else:
            cluster_input = pca_flat_clu
            cluster_space_tag = f"PCA residuals ({pca_flat_clu.shape[1]}-d)"
    elif args.cluster_on == "umap2d":
        cluster_input = emb2d
        cluster_space_tag = "2D UMAP projection"
    else:   # raw
        cluster_input = raw_flat_clu
        cluster_space_tag = f"raw residuals ({raw_flat_clu.shape[1]}-d)"
    print(f"   Clustering feature space: {cluster_space_tag}")

    # Determine the full (non-subsampled) cluster feature space used after
    # clustering to extend labels to all n_kept patches, so the spatial map
    # and cluster_purity probe cover the whole solar disk.
    if umap_sel is not None:
        if args.cluster_on == "pca" and pca_flat_clu_full is not None:
            cluster_input_full = pca_flat_clu_full
        elif args.cluster_on == "raw":
            cluster_input_full = raw_flat_clu_full
        else:  # umap2d: coords unavailable for non-sampled points; fall back
            cluster_input_full = (pca_flat_clu_full if pca_flat_clu_full is not None
                                  else raw_flat_clu_full)
    else:
        cluster_input_full = cluster_input  # no subsampling, already complete

    # --- Run the chosen clustering algorithm -------------------------------
    print(f"   Clustering ({args.cluster_method})")
    cluster_info: dict = {}
    cluster_labels_with_noise = None   # only populated for hdbscan
    if args.cluster_method == "hdbscan":
        labels_with_noise, labels_assigned, cluster_info = (
            _hdbscan_with_mahalanobis_reassignment(
                cluster_input,
                min_cluster_size = args.hdbscan_min_cluster_size,
                min_samples      = args.hdbscan_min_samples,
                cluster_selection_method  = args.hdbscan_cluster_selection_method,
                cluster_selection_epsilon = args.hdbscan_cluster_selection_epsilon,
                alpha            = args.hdbscan_alpha,
                mahalanobis_shrinkage = args.mahalanobis_shrinkage,
                reassign_noise   = args.mahalanobis_reassign,
            )
        )
        clu_labels_filt        = labels_assigned        # length n_kept
        clu_labels_filt_noise  = labels_with_noise      # length n_kept
        K_real = cluster_info.get("n_clusters", 0)
        print(f"   final K = {K_real} clusters "
              f"(noise frac before reassign = {cluster_info.get('noise_frac', 0):.3f})")
    elif args.cluster_method == "kmeans":
        from cuml.cluster import KMeans
        print(f"   k-means n_clusters={args.cluster_k}")
        km = KMeans(n_clusters=args.cluster_k, random_state=args.seed,
                    max_iter=300)
        clu_labels_filt = km.fit_predict(cluster_input).astype(np.int32)
        clu_labels_filt_noise = None
    else:   # gmm
        from sklearn.mixture import GaussianMixture
        print(f"   GMM n_components={args.cluster_k}")
        rng_fit = np.random.default_rng(args.seed)
        n_fit = min(200_000, len(cluster_input))
        fit_idx = rng_fit.choice(len(cluster_input), n_fit, replace=False)
        gmm = GaussianMixture(
            n_components=args.cluster_k, covariance_type="full",
            random_state=args.seed, max_iter=200, n_init=3,
        )
        gmm.fit(cluster_input[fit_idx])
        clu_labels_filt = gmm.predict(cluster_input).astype(np.int32)
        clu_labels_filt_noise = None

    # --- Extend cluster labels to ALL n_kept patches -------------------------
    # When --umap-n-points subsampled the data, clu_labels_filt only covers
    # the UMAP-sampled subset.  Extend to all n_kept patches so the spatial
    # map and cluster_purity probe see a fully coloured solar disk.
    if umap_sel is None:
        clu_labels_all = clu_labels_filt   # no subsampling: already complete
    elif args.cluster_method == "kmeans":
        clu_labels_all = km.predict(cluster_input_full).astype(np.int32)
        print(f"   k-means label extension: predicted {n_kept:,} patches")
    elif args.cluster_method == "gmm":
        clu_labels_all = gmm.predict(cluster_input_full).astype(np.int32)
        print(f"   GMM label extension: predicted {n_kept:,} patches")
    else:   # hdbscan: UMAP-sampled points keep their Mahalanobis-reassigned
            # labels; remaining patches are assigned via the same metric.
        clu_labels_all = np.full(n_kept, -1, dtype=np.int32)
        clu_labels_all[umap_sel] = clu_labels_filt
        non_umap_mask_ext = np.ones(n_kept, dtype=bool)
        non_umap_mask_ext[umap_sel] = False
        non_umap_idx = np.where(non_umap_mask_ext)[0]
        if len(non_umap_idx) > 0 and K_real > 0:
            # Build model from all UMAP-sampled points using post-reassignment
            # labels — covariances reflect the full (reassigned) cluster shapes.
            uniq_m, centroids_m, chol_m, logdet_m, n_diag_m = (
                _build_cluster_mahalanobis_model(
                    cluster_input.astype(np.float32, copy=False),
                    clu_labels_filt,
                    args.mahalanobis_shrinkage,
                )
            )
            if n_diag_m:
                print(f"   label extension: {n_diag_m}/{len(uniq_m)} clusters "
                      f"used diagonal-Σ fallback.")
            X_non = cluster_input_full[non_umap_idx].astype(np.float32, copy=False)
            print(f"   HDBSCAN label extension: Mahalanobis assignment for "
                  f"{len(non_umap_idx):,} non-UMAP patches ...")
            assigned_ext = _mahalanobis_assign_chunks(
                X_non, uniq_m, centroids_m, chol_m, logdet_m,
            )
            clu_labels_all[non_umap_idx] = assigned_ext

    # --- Expand cluster labels back to full length -------------------------
    # -2 marks patches that were excluded from clustering (NA when
    # --exclude-na-from-clustering was set).  Downstream probes (cluster_purity,
    # spatial map) detect the sentinel and skip those points.
    EXCLUDED = -2
    cluster_labels = np.full(n_total, EXCLUDED, dtype=np.int32)
    cluster_labels[keep_mask] = clu_labels_all
    np.save(out_dir / "cluster_labels.npy", cluster_labels)
    if clu_labels_filt_noise is not None:
        cluster_labels_with_noise = np.full(n_total, EXCLUDED, dtype=np.int32)
        cluster_labels_with_noise[umap_keep_mask] = clu_labels_filt_noise
        np.save(out_dir / "cluster_labels_with_noise.npy", cluster_labels_with_noise)
    else:
        cluster_labels_with_noise = None

    # ----- Plots: UMAP scatter colored three ways --------------------------
    # Plots use the filtered arrays directly — emb2d is already only the
    # clustered points, so all UMAP scatters live in clustered-only space.
    flat_lbl_filt  = full_flat_lbl[umap_keep_mask]
    flat_rows_full = np.broadcast_to(ab.positions[None, :, 0],
                                       (ab.N, ab.n_keep)).reshape(-1)
    flat_cols_full = np.broadcast_to(ab.positions[None, :, 1],
                                       (ab.N, ab.n_keep)).reshape(-1)
    flat_rows = flat_rows_full[umap_keep_mask]
    flat_cols = flat_cols_full[umap_keep_mask]

    title_suffix = (f"\nResiduals — {ab.N} samples × {ab.n_keep:,} patches "
                    f"= {emb2d.shape[0]:,} points")

    # By mask class — uses filtered labels because emb2d is filtered.
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for cls_id, cls_name in sorted(MASK_CLASS_NAMES.items()):
        m = flat_lbl_filt == cls_id
        if not m.any(): continue
        ax.scatter(emb2d[m, 0], emb2d[m, 1], c=MASK_PALETTE[cls_id],
                    label=f"{cls_name} (n={m.sum():,})",
                    s=3, alpha=0.4, linewidths=0)
    ax.legend(title="SPoCA class", markerscale=3)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    title = f"UMAP by mask class{title_suffix}"
    if args.exclude_na_from_clustering:
        title += "\n(NA patches excluded from clustering)"
    ax.set_title(title)
    fig.tight_layout()
    _save_fig(fig, out_dir / "umap_by_mask_class.png")

    # By cluster — palette scales gracefully for any K.
    # clu_labels_filt has the same length as emb2d, so we use it directly.
    unique_clusters = sorted(set(int(c) for c in clu_labels_filt.tolist()))
    K_real = max([c for c in unique_clusters if c >= 0], default=-1) + 1
    palette = _cluster_palette(K_real)
    noise_color = np.array([0.75, 0.75, 0.75, 1.0])
    LEGEND_THRESHOLD = 20

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for cid in unique_clusters:
        m = clu_labels_filt == cid
        col = noise_color if cid < 0 else palette[cid]
        label = "Noise" if cid < 0 else f"Cluster {cid} (n={m.sum():,})"
        ax.scatter(emb2d[m, 0], emb2d[m, 1], c=[col],
                    label=label, s=3, alpha=0.4, linewidths=0)
    if K_real <= LEGEND_THRESHOLD:
        ax.legend(title=f"{args.cluster_method.upper()} cluster",
                  markerscale=3, fontsize=7)
    else:
        ax.text(0.02, 0.98, f"{K_real} clusters (legend suppressed)",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP by cluster{title_suffix}")
    fig.tight_layout()
    _save_fig(fig, out_dir / "umap_by_cluster.png")

    # If HDBSCAN, also plot the with-noise version so noise structure is
    # visible (the assigned version is the one the rest of the pipeline uses).
    if (clu_labels_filt_noise is not None
            and (clu_labels_filt_noise == -1).any()):
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        for cid in sorted(set(int(c) for c in clu_labels_filt_noise.tolist())):
            m = clu_labels_filt_noise == cid
            col = noise_color if cid < 0 else palette[cid]
            label = "Noise" if cid < 0 else f"Cluster {cid}"
            ax.scatter(emb2d[m, 0], emb2d[m, 1], c=[col],
                        label=label, s=3, alpha=0.4, linewidths=0)
        if K_real <= LEGEND_THRESHOLD:
            ax.legend(title="HDBSCAN cluster (incl. noise)",
                      markerscale=3, fontsize=7)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_title(f"UMAP — HDBSCAN with noise (pre-reassignment)"
                     f"{title_suffix}")
        fig.tight_layout()
        _save_fig(fig, out_dir / "umap_by_cluster_with_noise.png")

    # By position — should look ~random in color if ablation worked.
    # flat_rows/flat_cols already filtered to align with emb2d.
    hue = flat_cols / PATCH_GRID
    val = 1.0 - flat_rows / PATCH_GRID
    sat = np.full_like(hue, 0.8)
    rgba = mcolors.hsv_to_rgb(np.stack([hue, sat, val], axis=1))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(emb2d[:, 0], emb2d[:, 1], c=rgba, s=3, alpha=0.4, linewidths=0)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP colored by patch position{title_suffix}\n"
                 f"(diagnostic: smooth gradient = position not fully removed)")
    fig.tight_layout()
    _save_fig(fig, out_dir / "umap_by_position.png")

    # Spatial cluster map for last sample.  Uses the FULL cluster_labels
    # array (with -2 sentinel for excluded patches) so excluded NA positions
    # render in the background color rather than getting a cluster color.
    excluded_color = np.array([0.93, 0.93, 0.93, 1.0])  # light gray, same as bg
    last = cluster_labels.reshape(ab.N, ab.n_keep)[-1]
    img = np.full((PATCH_GRID, PATCH_GRID, 4), 0.93, dtype=float)
    img[..., 3] = 1.0
    for i, (r, c) in enumerate(zip(ab.positions[:, 0], ab.positions[:, 1])):
        cid = int(last[i])
        if cid == EXCLUDED:
            img[r, c] = excluded_color
        elif cid < 0:
            img[r, c] = noise_color
        else:
            img[r, c] = palette[cid]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.imshow(img, origin="upper", interpolation="nearest")
    ax.set_xlabel("Patch column"); ax.set_ylabel("Patch row")
    title = f"Spatial cluster map (last sample, K={K_real})"
    if args.exclude_na_from_clustering:
        title += "\n(NA patches in light gray)"
    ax.set_title(title)
    fig.tight_layout()
    _save_fig(fig, out_dir / "spatial_cluster_labels.png")

    return {"cluster_labels": cluster_labels, "umap_2d": emb2d,
            "method": args.cluster_method, "k": K_real,
            "cluster_on": args.cluster_on,
            "cluster_space_tag": cluster_space_tag,
            "exclude_na": args.exclude_na_from_clustering,
            "n_clustered": int(keep_mask.sum()),
            "n_excluded": int((~keep_mask).sum()),
            "info": cluster_info}


# ============================================================================
# PROBE 7 — CLUSTER PURITY VS SPOCA
# ============================================================================
def probe_cluster_purity(cluster_labels: np.ndarray, mask_labels: np.ndarray,
                          args: argparse.Namespace, out_dir: Path) -> dict:
    """Cluster vs SPoCA confusion + purity metrics.

    Robust to large K (HDBSCAN can yield hundreds of clusters with
    min_cluster_size=15 on token-scale data).  When K exceeds
    --cluster-purity-top-n, the heatmap displays only the top-N largest
    clusters (sorted by size).  The full confusion matrix and per-cluster
    purity vector are always saved to JSON.
    """
    print("\n[probe 7/8] Cluster purity vs SPoCA")
    out_dir.mkdir(parents=True, exist_ok=True)

    flat_lbl = mask_labels.reshape(-1)

    # Filter out patches that were excluded from clustering entirely
    # (-2 sentinel set by probe_umap_cluster when --exclude-na-from-clustering
    # was on).  These patches are not part of the clustering universe and
    # would distort both the SPoCA-class breakdown and the purity numerator.
    in_clustering = cluster_labels != -2
    n_excluded_from_clustering = int((~in_clustering).sum())
    if n_excluded_from_clustering > 0:
        print(f"   {n_excluded_from_clustering:,} patches were excluded from "
              f"clustering (sentinel -2); restricting purity analysis to the "
              f"{int(in_clustering.sum()):,} clustered patches.")
        flat_lbl       = flat_lbl[in_clustering]
        cluster_labels = cluster_labels[in_clustering]

    # Real (non-noise) cluster IDs only.  cluster_labels may contain -1 if
    # HDBSCAN was used without --mahalanobis-reassignment.
    real_ids = sorted({int(c) for c in cluster_labels.tolist() if c >= 0})
    K = len(real_ids)
    classes = sorted(set(flat_lbl.tolist()))
    class_idx = {c: i for i, c in enumerate(classes)}

    # Build a contiguous index (0..K-1) over the real cluster IDs and report
    # the noise contingent separately so it doesn't fragment the heatmap.
    id_to_row = {cid: i for i, cid in enumerate(real_ids)}
    cm = np.zeros((K, len(classes)), dtype=np.int64)
    for cid, row in id_to_row.items():
        m = cluster_labels == cid
        for c in classes:
            cm[row, class_idx[c]] = int(((flat_lbl == c) & m).sum())

    cluster_sizes = cm.sum(axis=1)
    purity_per_cluster = np.where(
        cluster_sizes > 0,
        cm.max(axis=1) / cluster_sizes.clip(min=1),
        0.0,
    )
    weighted_purity = float(
        (purity_per_cluster * cluster_sizes).sum() / max(cluster_sizes.sum(), 1)
    )

    # ----- Physical (non-NA) purity decomposition --------------------------
    # The standard `weighted_purity` is dominated by the NA mega-cluster
    # whenever NA is in scope, which obscures on-disk structure.  We compute
    # three complementary views so the report can surface what's actually
    # interesting:
    #   1. weighted_purity                  (above; standard, NA-inclusive)
    #   2. weighted_purity_non_na_dominant  (over clusters whose argmax ≠ NA)
    #   3. weighted_physical_purity         (purity computed on the non-NA
    #                                        members of each cluster, weighted
    #                                        by physical_count)
    # Plus a "hidden physical" detector for clusters that are NA-dominant but
    # whose non-NA members agree on a single on-disk class — candidates for
    # SPoCA threshold misses (limb features, prominences, jets, etc.).

    NA_CLASS = -1
    na_idx = class_idx.get(NA_CLASS, None)

    if na_idx is not None and len(classes) > 1:
        na_counts        = cm[:, na_idx].astype(np.int64)
        physical_counts  = (cluster_sizes - na_counts).astype(np.int64)
        na_frac_per_cluster = np.where(
            cluster_sizes > 0, na_counts / cluster_sizes.clip(min=1), 0.0,
        )
        non_na_cols = [i for i in range(len(classes)) if i != na_idx]
        cm_non_na   = cm[:, non_na_cols]
        physical_max = cm_non_na.max(axis=1)
        physical_purity_per_cluster = np.where(
            physical_counts > 0,
            physical_max / physical_counts.clip(min=1),
            0.0,
        )
        physical_dominant_idx = cm_non_na.argmax(axis=1)
        physical_dominant_class = [
            classes[non_na_cols[i]] if physical_counts[k] > 0 else None
            for k, i in enumerate(physical_dominant_idx)
        ]
    else:
        # No NA in scope (probably --exclude-na-from-clustering was on).
        # Physical metrics reduce to the standard ones.
        na_counts = np.zeros(K, dtype=np.int64)
        physical_counts = cluster_sizes.copy()
        na_frac_per_cluster = np.zeros(K, dtype=np.float64)
        physical_purity_per_cluster = purity_per_cluster.copy()
        if K > 0:
            physical_dominant_class = [classes[i] for i in cm.argmax(axis=1)]
        else:
            physical_dominant_class = []

    # Per-cluster overall dominant class (standard argmax, NA-inclusive)
    if K > 0:
        dominant_class_per_cluster = [classes[i] for i in cm.argmax(axis=1)]
    else:
        dominant_class_per_cluster = []
    is_na_dominant = np.array(
        [c == NA_CLASS for c in dominant_class_per_cluster], dtype=bool
    )
    n_na_dominant     = int(is_na_dominant.sum())
    n_non_na_dominant = K - n_na_dominant

    # Aggregate metrics
    if n_non_na_dominant > 0:
        nna_sizes = cluster_sizes[~is_na_dominant]
        nna_purs  = purity_per_cluster[~is_na_dominant]
        weighted_purity_non_na_dominant = float(
            (nna_purs * nna_sizes).sum() / max(int(nna_sizes.sum()), 1)
        )
        n_points_in_non_na_dominant = int(nna_sizes.sum())
    else:
        weighted_purity_non_na_dominant = float("nan")
        n_points_in_non_na_dominant = 0

    total_physical = int(physical_counts.sum())
    if total_physical > 0:
        weighted_physical_purity = float(
            (physical_purity_per_cluster * physical_counts).sum() / total_physical
        )
    else:
        weighted_physical_purity = float("nan")

    # "Hidden physical" candidates: NA-dominant overall but non-NA members
    # agree strongly on a single on-disk class.  Tunable thresholds.
    HIDDEN_PURITY_THRESH   = 0.70
    HIDDEN_PHYSICAL_FRAC_MIN = 0.05   # require at least 5% non-NA members
    hidden_mask = (
        is_na_dominant
        & (physical_purity_per_cluster >= HIDDEN_PURITY_THRESH)
        & (1.0 - na_frac_per_cluster >= HIDDEN_PHYSICAL_FRAC_MIN)
    )
    n_hidden = int(hidden_mask.sum())

    # Noise stats (if any noise points are present)
    n_noise = int((cluster_labels < 0).sum())

    # Inverse normalization: per-class, what fraction of class members landed
    # in each cluster.
    class_total = cm.sum(axis=0)
    cm_class_norm = cm / class_total.clip(min=1)[None, :]
    cm_cluster_norm = cm / cluster_sizes.clip(min=1)[:, None]

    # Cluster size distribution stats — useful when K is large.
    if K > 0:
        size_quartiles = np.quantile(cluster_sizes,
                                       [0.25, 0.5, 0.75]).astype(int).tolist()
        size_min = int(cluster_sizes.min())
        size_max = int(cluster_sizes.max())
    else:
        size_quartiles = [0, 0, 0]
        size_min = size_max = 0

    results = {
        "K": K,
        "cluster_ids":          real_ids,
        "cluster_sizes":        cluster_sizes.tolist(),
        "spoca_classes":        classes,
        "spoca_class_names":    [MASK_CLASS_NAMES.get(c, str(c)) for c in classes],
        "confusion":            cm.tolist(),
        "row_normalized":       cm_cluster_norm.tolist(),
        "col_normalized":       cm_class_norm.tolist(),
        "purity_per_cluster":   purity_per_cluster.tolist(),
        "weighted_purity":      weighted_purity,
        # New physical-purity metrics
        "weighted_purity_non_na_dominant": weighted_purity_non_na_dominant,
        "weighted_physical_purity":        weighted_physical_purity,
        "physical_purity_per_cluster":     physical_purity_per_cluster.tolist(),
        "na_frac_per_cluster":             na_frac_per_cluster.tolist(),
        "physical_count_per_cluster":      physical_counts.tolist(),
        "dominant_class_per_cluster":      dominant_class_per_cluster,
        "physical_dominant_class_per_cluster":
            [None if c is None else int(c) for c in physical_dominant_class],
        "n_na_dominant":                   n_na_dominant,
        "n_non_na_dominant":               n_non_na_dominant,
        "n_points_in_non_na_dominant":     n_points_in_non_na_dominant,
        "n_hidden_physical_candidates":    n_hidden,
        "hidden_purity_threshold":         HIDDEN_PURITY_THRESH,
        "hidden_physical_frac_min":        HIDDEN_PHYSICAL_FRAC_MIN,
        # End new metrics
        "n_noise":              n_noise,
        "noise_frac":           n_noise / max(len(cluster_labels), 1),
        "n_excluded_from_clustering": n_excluded_from_clustering,
        "cluster_size_min":     size_min,
        "cluster_size_max":     size_max,
        "cluster_size_quartiles": size_quartiles,
    }

    print(f"   K = {K} clusters, weighted purity = {weighted_purity:.4f}")
    print(f"     ├─ NA-inclusive (standard)         : {weighted_purity:.4f}")
    print(f"     ├─ over non-NA-dominant clusters   : "
          f"{weighted_purity_non_na_dominant:.4f}  "
          f"({n_non_na_dominant}/{K} clusters, "
          f"{n_points_in_non_na_dominant:,} points)")
    print(f"     └─ physical purity (non-NA members): "
          f"{weighted_physical_purity:.4f}  "
          f"({total_physical:,} non-NA points across all clusters)")
    if n_hidden > 0:
        print(f"   {n_hidden} hidden-physical candidate clusters "
              f"(NA-dominant overall but ≥{int(100*HIDDEN_PURITY_THRESH)}% "
              f"physical purity).  See cluster_purity_hidden_physical.json.")
    if n_noise:
        print(f"   noise points present in label array: {n_noise:,} "
              f"({100*results['noise_frac']:.1f}%)")
    print(f"   cluster size: min={size_min}, q1={size_quartiles[0]}, "
          f"median={size_quartiles[1]}, q3={size_quartiles[2]}, max={size_max}")

    # Console listing — full when K is small, top + tail when K is large.
    LIST_FULL_THRESHOLD = 30
    sort_idx_desc = np.argsort(-cluster_sizes)
    if K <= LIST_FULL_THRESHOLD:
        listing = sort_idx_desc
    else:
        head = sort_idx_desc[:10]
        tail = sort_idx_desc[-3:]
        listing = list(head) + ["…"] + list(tail)
    for entry in listing:
        if entry == "…":
            print(f"   ... ({K - 13} clusters omitted) ...")
            continue
        i = int(entry)
        cid = real_ids[i]
        dom_idx = int(np.argmax(cm[i])) if cluster_sizes[i] else 0
        dom_name = MASK_CLASS_NAMES.get(classes[dom_idx], str(classes[dom_idx]))
        print(f"   cluster {cid:>4d}: n={cluster_sizes[i]:>9,}  "
              f"purity={purity_per_cluster[i]:.3f}  dominant={dom_name}")

    # ----- Plotting --------------------------------------------------------
    # When K > top-N, restrict the heatmap to the top-N largest clusters and
    # add a summary row that aggregates everything else.
    top_n = max(1, args.cluster_purity_top_n)
    show_aggregate_row = K > top_n
    n_rows = min(K, top_n) + (1 if show_aggregate_row else 0)
    sort_idx = sort_idx_desc[:min(K, top_n)]

    if show_aggregate_row:
        rest_idx = sort_idx_desc[top_n:]
        rest_cm  = cm[rest_idx].sum(axis=0)
        plot_cm  = np.vstack([cm[sort_idx], rest_cm[None, :]])
        plot_sizes = np.concatenate([cluster_sizes[sort_idx],
                                      [rest_cm.sum()]])
    else:
        plot_cm    = cm[sort_idx]
        plot_sizes = cluster_sizes[sort_idx]

    plot_cm_row = plot_cm / plot_sizes.clip(min=1)[:, None]
    # Column normalization uses the *full* class totals so the meaning
    # stays consistent for the aggregate row.
    plot_cm_col = plot_cm / class_total.clip(min=1)[None, :]

    row_labels = []
    for j, idx in enumerate(sort_idx):
        cid = real_ids[int(idx)]
        row_labels.append(f"C{cid} (n={int(cluster_sizes[idx]):,})")
    if show_aggregate_row:
        row_labels.append(f"all others (K={K - top_n}, n={int(rest_cm.sum()):,})")

    # Adapt figure height to the number of displayed rows so cells stay legible.
    fig_h = max(5, 0.32 * n_rows + 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(14, fig_h))
    for ax, (mat, title) in zip(axes, [
        (plot_cm_row, "Row-normalized: P(class | cluster)"),
        (plot_cm_col, "Col-normalized: P(cluster | class)"),
    ]):
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([MASK_CLASS_NAMES.get(c, str(c)) for c in classes],
                            rotation=30, ha="right")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_xlabel("SPoCA class"); ax.set_ylabel("Cluster (sorted by size)")
        ax.set_title(title)
        # Cell labels only when the heatmap isn't too crowded.
        if n_rows <= 20:
            for i in range(n_rows):
                for j in range(len(classes)):
                    txt = f"{mat[i,j]:.2f}"
                    ax.text(j, i, txt, ha="center", va="center",
                            color="white" if mat[i,j] > 0.5 else "black",
                            fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    title = (f"Cluster vs SPoCA confusion (K={K}, "
             f"weighted purity = {weighted_purity:.3f})")
    if show_aggregate_row:
        title += f"  —  showing top {top_n} clusters + aggregate"
    fig.suptitle(title)
    fig.tight_layout()
    _save_fig(fig, out_dir / "cluster_purity.png")

    # Cluster size distribution plot — informative when K is large.
    if K >= 5:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(np.arange(K), np.sort(cluster_sizes)[::-1],
               color="steelblue", edgecolor="none", width=1.0)
        ax.set_yscale("log")
        ax.set_xlabel("Cluster rank (largest first)")
        ax.set_ylabel("Cluster size (log)")
        ax.set_title(f"Cluster size distribution (K={K})")
        ax.grid(alpha=0.3, axis="y", which="both")
        fig.tight_layout()
        _save_fig(fig, out_dir / "cluster_size_distribution.png")

    # ----- NA-fraction × physical-purity atlas -----------------------------
    # Each point = one cluster.  X = NA fraction (1 means cluster is all NA;
    # 0 means cluster has no NA).  Y = physical purity (1 means the non-NA
    # members agree on a single on-disk class).  Point size ∝ cluster size.
    # Color = the cluster's *physical* dominant class.  This is the primary
    # visualization for finding "hidden physical" clusters: they live in the
    # top-right region (high NA fraction, high physical purity, with size
    # not negligible).
    if K > 0 and na_idx is not None:
        # Color clusters by their physical dominant class
        phys_dom_int = np.array(
            [-99 if c is None else int(c) for c in physical_dominant_class],
            dtype=np.int64,
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        # Faint vertical line at na_frac=0.5 (NA-dominant boundary by frac)
        ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5,
                   label="50% NA boundary")
        ax.axhline(HIDDEN_PURITY_THRESH, color="red", linestyle=":", alpha=0.5,
                   label=f"hidden-physical purity threshold "
                          f"({HIDDEN_PURITY_THRESH:.2f})")
        # Plot each physical class with its SPoCA color
        for cls_id, cls_name in sorted(MASK_CLASS_NAMES.items()):
            if cls_id == NA_CLASS:
                continue
            m = phys_dom_int == cls_id
            if not m.any():
                continue
            sizes = np.clip(cluster_sizes[m] / max(int(cluster_sizes.max()), 1) * 400,
                              4, 400)
            ax.scatter(na_frac_per_cluster[m], physical_purity_per_cluster[m],
                       s=sizes, c=MASK_PALETTE.get(cls_id, "gray"),
                       edgecolors="black", linewidths=0.3, alpha=0.6,
                       label=f"physical dom = {cls_name}")
        # Highlight hidden-physical candidates with a ring
        if hidden_mask.any():
            ax.scatter(na_frac_per_cluster[hidden_mask],
                       physical_purity_per_cluster[hidden_mask],
                       s=50, marker="o", facecolors="none",
                       edgecolors="red", linewidths=1.5,
                       label=f"hidden physical (n={int(hidden_mask.sum())})")
        ax.set_xlabel("NA fraction within cluster")
        ax.set_ylabel("Physical purity (within non-NA members)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Cluster atlas — NA fraction × physical purity (K={K})\n"
                     f"point size ∝ cluster total size; "
                     f"hidden-physical candidates ringed in red")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _save_fig(fig, out_dir / "cluster_atlas_na_vs_physical_purity.png")

    # ----- Hidden-physical detail file -------------------------------------
    # Sort hidden candidates by physical_count (largest non-NA contribution
    # first — those are the most likely to be real SPoCA misses).
    if n_hidden > 0:
        hidden_idx = np.where(hidden_mask)[0]
        hidden_sort = hidden_idx[np.argsort(-physical_counts[hidden_idx])]
        hidden_records = []
        for idx in hidden_sort:
            cid = real_ids[int(idx)]
            phys_dom_id = physical_dominant_class[int(idx)]
            hidden_records.append({
                "cluster_id": int(cid),
                "size_total": int(cluster_sizes[idx]),
                "size_na":    int(na_counts[idx]),
                "size_physical": int(physical_counts[idx]),
                "na_frac":    float(na_frac_per_cluster[idx]),
                "physical_purity": float(physical_purity_per_cluster[idx]),
                "physical_dominant_class":
                    None if phys_dom_id is None else int(phys_dom_id),
                "physical_dominant_class_name":
                    MASK_CLASS_NAMES.get(phys_dom_id, str(phys_dom_id))
                    if phys_dom_id is not None else None,
                "confusion_row": cm[idx].tolist(),
            })
        (out_dir / "cluster_purity_hidden_physical.json").write_text(
            json.dumps({
                "hidden_purity_threshold": HIDDEN_PURITY_THRESH,
                "hidden_physical_frac_min": HIDDEN_PHYSICAL_FRAC_MIN,
                "spoca_class_names":
                    [MASK_CLASS_NAMES.get(c, str(c)) for c in classes],
                "n_candidates": n_hidden,
                "candidates": hidden_records,
            }, indent=2)
        )

    (out_dir / "cluster_purity.json").write_text(
        json.dumps(_safe_json(results), indent=2)
    )
    return results


# ============================================================================
# PROBE 7b — SPATIAL DISTRIBUTION OF HIDDEN-PHYSICAL CANDIDATES
# ============================================================================
def probe_hidden_physical_spatial(
    cluster_labels: np.ndarray,
    ab: AblationOutputs,
    cluster_purity_results: dict,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    """Spatial visualization of hidden-physical cluster candidates.

    Renders the patch positions of clusters flagged as hidden-physical
    candidates (NA-dominant overall but with strong agreement on a single
    on-disk class among their non-NA members), so the user can visually
    inspect where these patches sit and check against expectations.

    Outputs in <out_dir>/:
      hidden_physical_density.png            spatial heatmap, all classes
      hidden_physical_density_per_class.png  3-panel split by physical-dom class
      hidden_physical_per_sample.png         per-sample maps (first 4 samples)
      hidden_physical_last_sample.png        full-resolution view of last sample
      hidden_physical_spatial.json           summary numbers
    """
    print("\n[probe 7b] Spatial distribution of hidden-physical candidates")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Identify hidden cluster IDs from the cluster_purity results ------
    real_ids       = cluster_purity_results["cluster_ids"]
    dom_per        = cluster_purity_results["dominant_class_per_cluster"]
    phys_pur_per   = cluster_purity_results["physical_purity_per_cluster"]
    phys_dom_per   = cluster_purity_results["physical_dominant_class_per_cluster"]
    na_frac_per    = cluster_purity_results["na_frac_per_cluster"]
    purity_thresh  = cluster_purity_results.get("hidden_purity_threshold", 0.7)
    frac_min       = cluster_purity_results.get("hidden_physical_frac_min", 0.05)

    NA_CLASS = -1
    hidden_id_to_phys_dom: dict[int, int] = {}
    for i, cid in enumerate(real_ids):
        if (dom_per[i] == NA_CLASS
                and phys_pur_per[i] >= purity_thresh
                and (1.0 - na_frac_per[i]) >= frac_min
                and phys_dom_per[i] is not None):
            hidden_id_to_phys_dom[int(cid)] = int(phys_dom_per[i])

    n_hidden = len(hidden_id_to_phys_dom)
    if n_hidden == 0:
        print("   no hidden-physical candidates to visualize "
              "(was --exclude-na-from-clustering on?  or no candidates "
              "passed the thresholds in cluster_purity)")
        results = {"n_hidden_clusters": 0}
        (out_dir / "hidden_physical_spatial.json").write_text(
            json.dumps(_safe_json(results), indent=2)
        )
        return results

    print(f"   visualizing {n_hidden} hidden-physical clusters")

    # ---- Vectorized hidden-mask computation -------------------------------
    rows = ab.positions[:, 0].astype(np.int64)         # (n_keep,)
    cols = ab.positions[:, 1].astype(np.int64)
    flat_rows = np.broadcast_to(rows[None, :], (ab.N, ab.n_keep)).reshape(-1)
    flat_cols = np.broadcast_to(cols[None, :], (ab.N, ab.n_keep)).reshape(-1)

    # phys_dom_lookup[cid] = physical-dominant class, or -1 if not hidden
    max_cid = max(hidden_id_to_phys_dom.keys())
    phys_dom_lookup = np.full(max_cid + 2, -1, dtype=np.int32)
    for cid, pd in hidden_id_to_phys_dom.items():
        phys_dom_lookup[cid] = pd

    flat_cluster = cluster_labels.astype(np.int64)
    # Patches whose cluster is in hidden set:
    is_hidden = np.zeros_like(flat_cluster, dtype=bool)
    cid_in_range = (flat_cluster >= 0) & (flat_cluster <= max_cid)
    is_hidden[cid_in_range] = phys_dom_lookup[flat_cluster[cid_in_range]] >= 0

    # ---- Build density grids ---------------------------------------------
    hidden_density = np.zeros((PATCH_GRID, PATCH_GRID), dtype=np.int64)
    np.add.at(hidden_density, (flat_rows[is_hidden], flat_cols[is_hidden]), 1)

    hidden_density_per_class: dict[int, np.ndarray] = {
        c: np.zeros((PATCH_GRID, PATCH_GRID), dtype=np.int64)
        for c in (0, 1, 2)
    }
    for cls_id in (0, 1, 2):
        mask = is_hidden & (phys_dom_lookup[flat_cluster.clip(0, max_cid)] == cls_id)
        np.add.at(hidden_density_per_class[cls_id],
                  (flat_rows[mask], flat_cols[mask]), 1)

    # ---- Plot 1: overall density heatmap ----------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    log_density = np.log1p(hidden_density.astype(np.float64))
    vmax = float(log_density.max()) if log_density.max() > 0 else 1.0
    im = ax.imshow(log_density, origin="upper", cmap="viridis",
                    vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_xlim(-0.5, PATCH_GRID - 0.5)
    ax.set_ylim(PATCH_GRID - 0.5, -0.5)
    plt.colorbar(im, ax=ax, label="log(1 + count)", fraction=0.046, pad=0.02)
    ax.set_title(
        f"Hidden-physical patches — spatial density across {ab.N} samples\n"
        f"({n_hidden} clusters, total = {int(hidden_density.sum()):,} patches)"
    )
    ax.set_xlabel("Patch column"); ax.set_ylabel("Patch row")
    fig.tight_layout()
    _save_fig(fig, out_dir / "hidden_physical_density.png")

    # ---- Plot 2: per-physical-class density panels ------------------------
    n_classes_with_data = sum(
        1 for c in (0, 1, 2) if hidden_density_per_class[c].sum() > 0
    )
    if n_classes_with_data > 0:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                                   sharex=True, sharey=True)
        for ax, cls_id in zip(axes, (0, 1, 2)):
            density = hidden_density_per_class[cls_id]
            log_d = np.log1p(density.astype(np.float64))
            local_vmax = max(float(log_d.max()), 0.5)
            im = ax.imshow(log_d, origin="upper", cmap="viridis",
                            vmin=0, vmax=local_vmax, interpolation="nearest")
            n_clusters_this_class = sum(
                1 for cid, pd in hidden_id_to_phys_dom.items() if pd == cls_id
            )
            ax.set_title(
                f"{MASK_CLASS_NAMES[cls_id]}\n"
                f"({n_clusters_this_class} clusters, "
                f"{int(density.sum()):,} patches)"
            )
            ax.set_xlabel("Patch column")
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        axes[0].set_ylabel("Patch row")
        fig.suptitle("Hidden-physical density — split by physical-dominant class")
        fig.tight_layout()
        _save_fig(fig, out_dir / "hidden_physical_density_per_class.png")

    # ---- Plot 3: per-sample maps (first 4 samples) ------------------------
    n_show = min(4, ab.N)
    sample_indices = np.linspace(0, ab.N - 1, n_show, dtype=int)
    cluster_labels_2d = cluster_labels.reshape(ab.N, ab.n_keep)

    fig, axes = plt.subplots(2, 2, figsize=(11, 11),
                               sharex=True, sharey=True)
    axes_flat = axes.flatten()
    bg_color    = np.array([0.93, 0.93, 0.93, 1.0])
    on_disk_color = np.array([0.85, 0.85, 0.85, 1.0])
    for ax_idx, t in enumerate(sample_indices):
        ax = axes_flat[ax_idx]
        img = np.tile(bg_color, (PATCH_GRID, PATCH_GRID, 1))
        # Light gray for on-disk in this sample
        for i in range(ab.n_keep):
            r, c = int(rows[i]), int(cols[i])
            if ab.mask_labels[t, i] != -1:
                img[r, c] = on_disk_color
            cid = int(cluster_labels_2d[t, i])
            if cid in hidden_id_to_phys_dom:
                phys_dom = hidden_id_to_phys_dom[cid]
                color_hex = MASK_PALETTE.get(phys_dom, "#888888")
                img[r, c] = np.array(mcolors.to_rgba(color_hex))
        ax.imshow(img, origin="upper", interpolation="nearest")
        ax.set_title(f"Sample {t}")
        ax.set_xlim(-0.5, PATCH_GRID - 0.5)
        ax.set_ylim(PATCH_GRID - 0.5, -0.5)
    for ax_idx in range(n_show, 4):
        axes_flat[ax_idx].set_visible(False)
    # Build a manual legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=on_disk_color,
                                       label="on-disk (in this sample)")]
    for cls_id in (0, 1, 2):
        if hidden_density_per_class[cls_id].sum() > 0:
            legend_handles.append(plt.Rectangle(
                (0, 0), 1, 1, color=MASK_PALETTE[cls_id],
                label=f"hidden = {MASK_CLASS_NAMES[cls_id]}",
            ))
    fig.legend(handles=legend_handles, loc="lower center",
                ncol=len(legend_handles), fontsize=9,
                bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Per-sample hidden-physical patches "
                  "(first 4 samples, evenly spaced)")
    fig.tight_layout()
    _save_fig(fig, out_dir / "hidden_physical_per_sample.png")

    # ---- Plot 3b: full-resolution view of just the last sample -----------
    last_t = ab.N - 1
    img = np.tile(bg_color, (PATCH_GRID, PATCH_GRID, 1))

    on_disk_this = ab.mask_labels[last_t] != -1
    img[rows[on_disk_this], cols[on_disk_this]] = on_disk_color

    cluster_this = cluster_labels_2d[last_t]
    in_range = (cluster_this >= 0) & (cluster_this <= max_cid)
    phys_dom_this = np.full(ab.n_keep, -1, dtype=np.int32)
    phys_dom_this[in_range] = phys_dom_lookup[cluster_this[in_range]]
    for cls_id in (0, 1, 2):
        m = phys_dom_this == cls_id
        if m.any():
            color = np.array(mcolors.to_rgba(MASK_PALETTE[cls_id]))
            img[rows[m], cols[m]] = color

    # Resolve timestamp string (best-effort)
    last_ts_repr = None
    try:
        ts_val = ab.timestamps[last_t]
        if hasattr(ts_val, "isoformat"):
            last_ts_repr = ts_val.isoformat()
        else:
            last_ts_repr = str(ts_val)
    except (AttributeError, IndexError, TypeError):
        pass

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, origin="upper", interpolation="nearest")
    ax.set_xlim(-0.5, PATCH_GRID - 0.5)
    ax.set_ylim(PATCH_GRID - 0.5, -0.5)
    title = f"Hidden-physical patches — sample {last_t} (last sample of {ab.N})"
    if last_ts_repr is not None:
        title += f"\ntimestamp: {last_ts_repr}"
    ax.set_title(title)
    ax.set_xlabel("Patch column")
    ax.set_ylabel("Patch row")
    n_hidden_this = {
        cls_id: int((phys_dom_this == cls_id).sum()) for cls_id in (0, 1, 2)
    }
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=on_disk_color,
                                       label="on-disk (in this sample)")]
    for cls_id in (0, 1, 2):
        if n_hidden_this[cls_id] > 0:
            legend_handles.append(plt.Rectangle(
                (0, 0), 1, 1, color=MASK_PALETTE[cls_id],
                label=f"hidden = {MASK_CLASS_NAMES[cls_id]} "
                      f"(n={n_hidden_this[cls_id]:,})",
            ))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10,
               framealpha=0.85)
    fig.tight_layout()
    _save_fig(fig, out_dir / "hidden_physical_last_sample.png")
    print(f"   last sample (t={last_t}): "
          f"hidden patches by class — "
          f"CH={n_hidden_this[0]:,}, QS={n_hidden_this[1]:,}, "
          f"AR={n_hidden_this[2]:,}")

    # ---- Per-class summary counts -----------------------------------------
    summary_per_class: dict[str, dict] = {}
    for cls_id in (0, 1, 2):
        d = hidden_density_per_class[cls_id].flatten()
        if d.sum() == 0:
            continue
        summary_per_class[MASK_CLASS_NAMES[cls_id]] = {
            "n_clusters": sum(1 for cid, pd in hidden_id_to_phys_dom.items()
                              if pd == cls_id),
            "n_hidden_patches": int(d.sum()),
        }
        print(f"   {MASK_CLASS_NAMES[cls_id]:14s}: "
              f"{int(d.sum()):>7,} patches across "
              f"{summary_per_class[MASK_CLASS_NAMES[cls_id]]['n_clusters']} "
              f"clusters")

    results = {
        "n_hidden_clusters": n_hidden,
        "total_hidden_patches": int(hidden_density.sum()),
        "by_physical_class": summary_per_class,
        "hidden_purity_threshold": purity_thresh,
        "hidden_physical_frac_min": frac_min,
        "last_sample_index": int(last_t),
        "last_sample_timestamp": last_ts_repr,
        "last_sample_hidden_counts": {
            MASK_CLASS_NAMES[cls_id]: n_hidden_this[cls_id]
            for cls_id in (0, 1, 2)
        },
    }
    (out_dir / "hidden_physical_spatial.json").write_text(
        json.dumps(_safe_json(results), indent=2)
    )
    return results


# ============================================================================
# PROBE 8 — PCA RGB VISUALIZATION (with optional AnyUp)
# ============================================================================
def probe_pca_viz(raw: np.ndarray, residuals: np.ndarray,
                   pixels: np.ndarray, ab: AblationOutputs,
                   args: argparse.Namespace, out_dir: Path) -> dict:
    """Per-sample 3-component PCA RGB spatial map.

    Each patch (one of n_keep tokens per sample) is projected onto the first
    3 PCs of either the raw or residual embedding space.  The 3 scalar scores
    become R, G, B for that patch's pixel in the output map.

    Two normalization modes:

    Per-sample (--pca-viz-norm per_sample, default):
        Each sample's percentiles computed on its own data.  Within-sample
        structure is clearest, but a "very red" patch in sample 0 and "very
        red" patch in sample 5 do NOT have the same absolute PC1 score.
        Use this for spotting structure within individual snapshots.

    Global (--pca-viz-norm global):
        Percentiles computed across ALL samples (entire N×n_keep×3 score
        cube).  Same color across samples means the same (relative-to-
        global-distribution) PC score, so cross-sample comparisons are
        meaningful.  Use this for tracking how the embedding of a fixed
        feature evolves over time.  Within-sample contrast may be lower
        if one sample dominates the global distribution.

    Both (--pca-viz-norm both):
        Renders a 2×2 raw/residual × per-sample/global grid per sample
        (plus an input mean panel for context).  4 PCA RGB panels per
        sample.  Useful for the head-to-head comparison.

    Embedding spaces selected by --pca-viz-on (raw, residual, both).
    """
    print("\n[probe 8/8] PCA RGB visualization")
    from sklearn.decomposition import PCA
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    space_mode = getattr(args, "pca_viz_on", "both")
    norm_mode  = getattr(args, "pca_viz_norm", "per_sample")
    do_raw    = space_mode in ("raw", "both")
    do_res    = space_mode in ("residual", "both")
    do_per    = norm_mode  in ("per_sample", "both")
    do_global = norm_mode  in ("global", "both")

    spaces: list[str] = []
    if do_raw: spaces.append("raw")
    if do_res: spaces.append("residual")
    norms: list[str] = []
    if do_per:    norms.append("per_sample")
    if do_global: norms.append("global")

    # ---- Fit PCAs on a shared subsample of row indices --------------------
    n_total = residuals.shape[0] * residuals.shape[1]
    n_sub   = min(200_000, n_total)
    perm    = rng.permutation(n_total)[:n_sub]

    pca_dict: dict[str, PCA] = {}
    arr_dict: dict[str, np.ndarray] = {}
    if do_raw:
        flat_raw = raw.reshape(-1, raw.shape[-1])
        pca_dict["raw"] = PCA(n_components=3, random_state=args.seed,
                                svd_solver="randomized").fit(flat_raw[perm])
        arr_dict["raw"] = raw
        ev = pca_dict["raw"].explained_variance_ratio_
        print(f"   PCA on raw      — 3-comp EV: {ev.sum():.3f}  "
              f"(per-PC: {ev[0]:.3f}, {ev[1]:.3f}, {ev[2]:.3f})")
        np.save(out_dir / "pca_components_raw.npy", pca_dict["raw"].components_)
        np.save(out_dir / "pca_explained_variance_ratio_raw.npy", ev)
    if do_res:
        flat_res = residuals.reshape(-1, residuals.shape[-1])
        pca_dict["residual"] = PCA(n_components=3, random_state=args.seed,
                                      svd_solver="randomized").fit(flat_res[perm])
        arr_dict["residual"] = residuals
        ev = pca_dict["residual"].explained_variance_ratio_
        print(f"   PCA on residual — 3-comp EV: {ev.sum():.3f}  "
              f"(per-PC: {ev[0]:.3f}, {ev[1]:.3f}, {ev[2]:.3f})")
        np.save(out_dir / "pca_components_residual.npy",
                pca_dict["residual"].components_)
        np.save(out_dir / "pca_explained_variance_ratio_residual.npy", ev)

    # ---- Pre-transform all samples to enable global percentiles ----------
    # Memory cost: N × n_keep × 3 × 4 bytes (float32) per space.
    # For N=16, n_keep=65536: ~12.6 MB per space.  Negligible.
    all_scores: dict[str, np.ndarray] = {}
    for tag in spaces:
        all_scores[tag] = np.empty((ab.N, ab.n_keep, 3), dtype=np.float32)
        for t in range(ab.N):
            all_scores[tag][t] = pca_dict[tag].transform(arr_dict[tag][t])

    # Global percentiles per (space, component)
    global_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if do_global:
        for tag in spaces:
            flat = all_scores[tag].reshape(-1, 3)
            lo = np.percentile(flat, 2,  axis=0)
            hi = np.percentile(flat, 98, axis=0)
            global_bounds[tag] = (lo, hi)
            print(f"   global PC bounds ({tag}): "
                  f"PC1=[{lo[0]:+.2f}, {hi[0]:+.2f}], "
                  f"PC2=[{lo[1]:+.2f}, {hi[1]:+.2f}], "
                  f"PC3=[{lo[2]:+.2f}, {hi[2]:+.2f}]")

    # Optional AnyUp upsampler — applied to residual RGB only, per-sample norm
    upsampler = None
    if args.use_anyup and do_res:
        try:
            print("   Loading AnyUp from torch.hub ...")
            upsampler = torch.hub.load("wimmerth/anyup", "anyup")
            upsampler = upsampler.to(DEVICE).eval()
            print("   AnyUp loaded — will be applied to residual RGB "
                  "(per-sample norm)")
        except Exception as e:
            print(f"   AnyUp unavailable ({e}) — falling back to no-upsample")
            upsampler = None

    rows_idx = ab.positions[:, 0].astype(np.int64)
    cols_idx = ab.positions[:, 1].astype(np.int64)

    def _pnorm_per_sample(a: np.ndarray, lo: int = 2, hi: int = 98) -> np.ndarray:
        vl, vh = np.percentile(a, [lo, hi])
        return np.clip((a - vl) / (vh - vl + 1e-8), 0.0, 1.0)

    def _pnorm_with_bounds(a: np.ndarray, vl: float, vh: float) -> np.ndarray:
        return np.clip((a - vl) / (vh - vl + 1e-8), 0.0, 1.0)

    def _build_rgb(scores: np.ndarray, norm_tag: str,
                    space_tag: str) -> np.ndarray:
        rgb = np.full((PATCH_GRID, PATCH_GRID, 3), 0.5, dtype=np.float64)
        if norm_tag == "per_sample":
            for k in range(3):
                rgb[rows_idx, cols_idx, k] = _pnorm_per_sample(scores[:, k])
        else:  # global
            lo, hi = global_bounds[space_tag]
            for k in range(3):
                rgb[rows_idx, cols_idx, k] = _pnorm_with_bounds(
                    scores[:, k], lo[k], hi[k])
        return rgb

    n_show = min(args.pca_viz_n_samples, ab.N)
    sample_indices = np.linspace(0, ab.N - 1, n_show, dtype=int)
    print(f"   rendering {n_show} sample panels "
          f"(pca-viz-on={space_mode}, pca-viz-norm={norm_mode})")

    n_rows = max(1, len(norms))
    n_cols = 1 + len(spaces)  # +1 for the input column

    for s_i, t in enumerate(sample_indices):
        # Mean-channel input for the guide / thumbnail
        guide_full = pixels[t].mean(axis=-1)
        guide_grid = np.full((PATCH_GRID, PATCH_GRID), np.nan)
        guide_grid[rows_idx, cols_idx] = guide_full

        # Resolve timestamp string for this sample
        ts_repr = None
        try:
            ts_val = ab.timestamps[t]
            ts_repr = ts_val.isoformat() if hasattr(ts_val, "isoformat") else str(ts_val)
        except (AttributeError, IndexError, TypeError):
            pass
        ts_suffix = f"\n{ts_repr}" if ts_repr else ""

        fig, axes = plt.subplots(n_rows, n_cols,
                                   figsize=(4.5 * n_cols, 4.5 * n_rows),
                                   squeeze=False)

        # Input panel: top-left only.  If 2 rows, blank the bottom-left.
        axes[0, 0].imshow(_pnorm_per_sample(np.nan_to_num(guide_grid)),
                          cmap="gray")
        axes[0, 0].set_title(f"Sample {t} — input mean{ts_suffix}")
        axes[0, 0].axis("off")
        if n_rows > 1:
            axes[1, 0].axis("off")

        # PCA RGB panels: rows = norms, cols = spaces (offset by +1 for input)
        for r, norm_tag in enumerate(norms):
            for c, space_tag in enumerate(spaces):
                ax = axes[r, c + 1]
                rgb = _build_rgb(all_scores[space_tag][t], norm_tag, space_tag)
                ax.imshow(rgb)
                ev = pca_dict[space_tag].explained_variance_ratio_
                norm_label = ("per-sample" if norm_tag == "per_sample"
                               else "global")
                content_label = ("position+content" if space_tag == "raw"
                                  else "content-only")
                ax.set_title(f"PCA RGB ({space_tag}, {norm_label})\n"
                              f"3-PC EV = {ev.sum():.2f}  ({content_label})")
                ax.axis("off")

        # AnyUp panel (residual + per-sample only).  Append as an extra column
        # to the figure if requested — saves to a separate file rather than
        # forcing the main 2x2 layout to grow asymmetrically.
        fig.tight_layout()
        _save_fig(fig, out_dir / f"pca_viz_sample_{t:03d}.png")

        if upsampler is not None and do_res and do_per:
            try:
                # Reuse the residual per-sample scores (we already have them
                # in all_scores).
                residual_scores = all_scores["residual"][t]
                lr_grid = np.zeros((3, PATCH_GRID, PATCH_GRID), dtype=np.float32)
                for k in range(3):
                    lr_grid[k][rows_idx, cols_idx] = residual_scores[:, k]
                lr = torch.from_numpy(lr_grid[None]).float().to(DEVICE)
                guide = torch.from_numpy(np.nan_to_num(guide_grid)[None, None]
                                            ).float()
                guide = torch.nn.functional.interpolate(
                    guide, size=(args.anyup_resolution, args.anyup_resolution),
                    mode="bilinear", align_corners=False,
                ).expand(-1, 3, -1, -1).to(DEVICE)
                with torch.no_grad():
                    hr = upsampler(guide, lr, q_chunk_size=10)
                hr_rgb = hr[0].permute(1, 2, 0).cpu().numpy()
                hr_rgb = np.stack(
                    [_pnorm_per_sample(hr_rgb[..., k]) for k in range(3)],
                    axis=-1,
                )
                fig2, ax2 = plt.subplots(figsize=(7.0, 7.0))
                ax2.imshow(hr_rgb)
                ax2.set_title(f"Sample {t} — PCA RGB + AnyUp "
                               f"(residual basis, per-sample norm){ts_suffix}")
                ax2.axis("off")
                fig2.tight_layout()
                _save_fig(fig2, out_dir / f"pca_viz_sample_{t:03d}_anyup.png")
            except Exception as e:
                print(f"   AnyUp render failed for sample {t}: {e}")

    # Build summary results
    results: dict = {
        "n_samples_rendered": n_show,
        "pca_viz_on":  space_mode,
        "pca_viz_norm": norm_mode,
    }
    if do_raw:
        ev = pca_dict["raw"].explained_variance_ratio_
        results["explained_variance_ratio_3_raw"] = ev.tolist()
    if do_res:
        ev = pca_dict["residual"].explained_variance_ratio_
        results["explained_variance_ratio_3_residual"] = ev.tolist()
        # Backwards compat: legacy field name
        results["explained_variance_ratio_3"] = ev.tolist()
    if do_global:
        results["global_pc_bounds"] = {
            tag: {"lo": lo.tolist(), "hi": hi.tolist()}
            for tag, (lo, hi) in global_bounds.items()
        }
    return results


# ============================================================================
# REPORT WRITER
# ============================================================================
def write_unified_report(results: dict, ab: AblationOutputs,
                          args: argparse.Namespace, out_dir: Path) -> None:
    md_path = out_dir / "unified_report.md"
    lines = []
    lines.append("# Surya Unified Probing Report")
    lines.append("")
    lines.append(f"- ablation_dir   : `{args.ablation_dir}`")
    lines.append(f"- N_samples      : {ab.N}")
    lines.append(f"- n_keep_patches : {ab.n_keep}")
    lines.append(f"- embed_dim D    : {ab.embed_dim}")
    lines.append(f"- probes_run     : {args.probes}")
    lines.append("")
    lines.append("## Position-vs-content attribution")
    lines.append("")
    lines.append("Each probe runs once on raw embeddings and once on residuals "
                  "(post per-position mean ablation).  The delta isolates how "
                  "much of the metric is driven by Surya's Fourier position "
                  "encoding vs. genuine content.")
    lines.append("")

    if "effective_rank" in results:
        r = results["effective_rank"]
        lines.append("## Effective rank")
        lines.append("")
        lines.append("| stratum | raw | residual | Δ (position) |")
        lines.append("|---------|-----|----------|--------------|")
        for tag, v in r.items():
            lines.append(f"| {tag} | {v['raw']:.2f} | {v['residual']:.2f} "
                          f"| {v['delta_position']:+.2f} |")
        lines.append("")
        lines.append(f"![effective rank](effective_rank/effective_rank.png)")
        lines.append("")

    if "pca_ev" in results:
        r = results["pca_ev"]
        lines.append("## PCA explained variance")
        lines.append("")
        lines.append("| stratum | n→50% (raw) | n→50% (res) | n→90% (raw) | n→90% (res) |")
        lines.append("|---------|-------------|-------------|-------------|-------------|")
        for tag, v in r.items():
            lines.append(f"| {tag} | {v.get('raw_n_to_50pct')} | "
                          f"{v.get('residual_n_to_50pct')} | "
                          f"{v.get('raw_n_to_90pct')} | "
                          f"{v.get('residual_n_to_90pct')} |")
        lines.append("")
        lines.append(f"![PCA EV overall](pca_ev/pca_ev_overall.png)")
        lines.append(f"![PCA EV by class](pca_ev/pca_ev_by_class.png)")
        lines.append("")

    if "linear_probe_px" in results:
        r = results["linear_probe_px"]
        lines.append("## Linear probe — pixel reconstruction (per channel group)")
        lines.append("")
        cfg_bits = [f"CV scheme: {r['scheme']} ({r['k_splits']} splits)",
                    f"subsample={r['n_patches_subsample']}",
                    f"D={r.get('embed_dim', '?')}"]
        if r.get("use_pca_for_probe"):
            cfg_bits.append("**PCA-reduced inputs**")
        if r.get("alpha_sweep_active"):
            cfg_bits.append(f"alpha tuned by inner CV over {r['alphas_swept']}")
        else:
            a = r.get("alphas_swept", [None])[0]
            cfg_bits.append(f"alpha={a} (fixed)")
        lines.append(", ".join(cfg_bits))
        lines.append("")
        lines.append("| group | raw R² | residual R² | Δ |")
        lines.append("|-------|--------|-------------|---|")
        for g in CHANNEL_GROUPS:
            raw_g = r["raw"]["_group_r2"][g]
            res_g = r["residual"]["_group_r2"][g]
            lines.append(f"| {g} | {raw_g:+.3f} | {res_g:+.3f} | "
                          f"{raw_g - res_g:+.3f} |")
        lines.append("")
        if r.get("alpha_sweep_active"):
            lines.append("### Median chosen alpha per group")
            lines.append("")
            lines.append("| group | raw α (median) | residual α (median) |")
            lines.append("|-------|----------------|---------------------|")
            for g, idxs in CHANNEL_GROUPS.items():
                ra = np.median([r["raw"][CHANNELS[i]]["alpha_median"] for i in idxs])
                re = np.median([r["residual"][CHANNELS[i]]["alpha_median"] for i in idxs])
                lines.append(f"| {g} | {ra:g} | {re:g} |")
            lines.append("")
        lines.append(f"![linear probe pixels](linear_probe_px/linear_probe_px.png)")
        lines.append("")

    if "linear_probe_px_stratified" in results:
        r = results["linear_probe_px_stratified"]
        lines.append("## Linear probe — pixel reconstruction, stratified by SPoCA class")
        lines.append("")
        cfg_bits = [f"D = {r.get('embed_dim', '?')}",
                    f"per-(sample, class) cap = "
                    f"{r.get('n_per_class_per_sample_cap', '?')}"]
        if r.get("use_pca_for_probe"):
            cfg_bits.append("**PCA-reduced inputs**")
        if r.get("alpha_sweep_active"):
            cfg_bits.append(f"alpha tuned by inner CV over {r['alphas_swept']}")
        else:
            a = r.get("alphas_swept", [None])[0]
            cfg_bits.append(f"alpha={a} (fixed)")
        lines.append(", ".join(cfg_bits))
        lines.append("")

        cls_names_seen = list(r["by_class"].keys())
        if not cls_names_seen:
            lines.append("_No classes had enough valid samples; probe was a no-op._")
            lines.append("")
        else:
            # Compact summary table: rows = group, columns = (class, raw/res/Δ)
            header_top = "| group | " + " | ".join(
                [f"{cn} raw | {cn} res | Δ" for cn in cls_names_seen]
            ) + " |"
            sep = "|" + "|".join(
                ["---"] + ["---" for _ in range(3 * len(cls_names_seen))]
            ) + "|"
            lines.append(header_top)
            lines.append(sep)
            for g in CHANNEL_GROUPS:
                row = [g]
                for cn in cls_names_seen:
                    raw_v = r["by_class"][cn]["raw"]["_group_r2"][g]
                    res_v = r["by_class"][cn]["residual"]["_group_r2"][g]
                    row += [f"{raw_v:+.3f}", f"{res_v:+.3f}",
                            f"{raw_v - res_v:+.3f}"]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

            # Per-class effective N table
            lines.append("### Per-class effective sample sizes")
            lines.append("")
            lines.append("| class | n_valid_samples | avg patches/sample | CV scheme |")
            lines.append("|-------|-----------------|--------------------|-----------|")
            for cn in cls_names_seen:
                d = r["by_class"][cn]
                lines.append(f"| {cn} | {d['n_valid_samples']} "
                              f"| {d['avg_patches_per_sample']:.0f} "
                              f"| {d['scheme']} ({d['k_splits']} splits) |")
            lines.append("")

        lines.append(f"![stratified probe (raw vs residual)]"
                      f"(linear_probe_px_stratified/linear_probe_px_stratified.png)")
        lines.append(f"![stratified probe (Δ = position contribution)]"
                      f"(linear_probe_px_stratified/linear_probe_px_stratified_delta.png)")
        lines.append("")

    if "linear_probe_cls" in results:
        r = results["linear_probe_cls"]
        lines.append("## Linear probe — SPoCA class")
        lines.append("")
        cfg_bits = [f"CV scheme: {r['scheme']} ({r['k_splits']} splits)",
                    f"D={r.get('embed_dim', '?')}"]
        if r.get("use_pca_for_probe"):
            cfg_bits.append("**PCA-reduced inputs**")
        if r.get("alpha_sweep_active"):
            cfg_bits.append(f"C tuned by inner CV (alphas={r['alphas_swept']})")
        lines.append(", ".join(cfg_bits))
        lines.append("")
        lines.append("| variant | raw acc | residual acc | Δ | raw α (med) | res α (med) |")
        lines.append("|---------|---------|--------------|---|-------------|-------------|")
        for variant in ("4class", "3class"):
            raw_d = r["raw"].get(variant, {})
            res_d = r["residual"].get(variant, {})
            ra = raw_d.get("accuracy")
            re = res_d.get("accuracy")
            if ra is None or re is None:
                continue
            ra_alpha = raw_d.get("alpha_implied_median", "—")
            re_alpha = res_d.get("alpha_implied_median", "—")
            ra_alpha_s = f"{ra_alpha:g}" if isinstance(ra_alpha, (int, float)) else ra_alpha
            re_alpha_s = f"{re_alpha:g}" if isinstance(re_alpha, (int, float)) else re_alpha
            lines.append(f"| {variant} | {ra:.4f} | {re:.4f} | {ra - re:+.4f} | "
                          f"{ra_alpha_s} | {re_alpha_s} |")
        lines.append("")
        lines.append(f"![confusion matrices](linear_probe_cls/linear_probe_cls_confusion.png)")
        lines.append("")

    if "spatial_corr" in results:
        r = results["spatial_corr"]
        lines.append("## Spatial consistency")
        lines.append("")
        lines.append(f"- raw      Pearson(emb dist, grid dist) = "
                      f"{r['raw_mean']:.4f} ± {r['raw_std']:.4f}")
        lines.append(f"- residual Pearson(emb dist, grid dist) = "
                      f"{r['residual_mean']:.4f} ± {r['residual_std']:.4f}")
        lines.append(f"- Δ (position-driven spatial autocorrelation) = "
                      f"**{r['delta_mean']:+.4f}**")
        lines.append("")
        lines.append(f"![spatial correlation](spatial_corr/spatial_corr.png)")
        lines.append("")

    if "umap_cluster" in results and "cluster_purity" in results:
        r = results["cluster_purity"]
        u = results["umap_cluster"]
        lines.append("## UMAP + clustering — purity vs SPoCA")
        lines.append("")
        cfg_bits = [f"method = `{u.get('method', '?')}`",
                    f"clustered on = `{u.get('cluster_space_tag', '?')}`",
                    f"K = {r['K']}"]
        if u.get("exclude_na"):
            cfg_bits.append(f"NA excluded ({u.get('n_excluded', 0):,} of "
                             f"{u.get('n_clustered', 0) + u.get('n_excluded', 0):,})")
        if r.get("noise_frac", 0) > 0:
            cfg_bits.append(f"noise = {r['n_noise']:,} "
                             f"({100*r['noise_frac']:.1f}%)")
        lines.append("- " + ", ".join(cfg_bits))
        if u.get("method") == "hdbscan":
            info = u.get("info", {})
            if info:
                lines.append(f"- HDBSCAN: pre-reassignment noise frac = "
                              f"{info.get('noise_frac', 0):.3f}; "
                              f"diagonal-Σ fallbacks = "
                              f"{info.get('n_diag_fallback', 0)}/{info.get('n_clusters', 0)}")
        if r["K"] > 0:
            qs = r.get("cluster_size_quartiles", [0, 0, 0])
            lines.append(f"- Cluster size: min={r['cluster_size_min']}, "
                          f"q1={qs[0]}, median={qs[1]}, q3={qs[2]}, "
                          f"max={r['cluster_size_max']}")
        lines.append("")

        # Three-view purity summary — the headline finding.
        lines.append("### Purity views")
        lines.append("")
        lines.append("| metric | value | interpretation |")
        lines.append("|--------|-------|----------------|")
        lines.append(f"| weighted purity (NA-inclusive) "
                      f"| {r['weighted_purity']:.4f} "
                      f"| standard, dominated by NA mega-cluster |")
        wp_nna = r.get("weighted_purity_non_na_dominant", float("nan"))
        n_nna  = r.get("n_non_na_dominant", 0)
        n_pts_nna = r.get("n_points_in_non_na_dominant", 0)
        lines.append(f"| over non-NA-dominant clusters "
                      f"| {wp_nna:.4f} "
                      f"| **physical structure**: {n_nna}/{r['K']} clusters, "
                      f"{n_pts_nna:,} points |")
        wpp = r.get("weighted_physical_purity", float("nan"))
        lines.append(f"| weighted physical purity "
                      f"| {wpp:.4f} "
                      f"| how clean the on-disk content is within each "
                      f"cluster (weighted by non-NA member count) |")
        lines.append("")

        # Top-N table — always limit to keep the markdown readable.
        TOP_N_TABLE = 30
        sort_idx = sorted(range(r["K"]),
                           key=lambda i: -r["cluster_sizes"][i])[:TOP_N_TABLE]
        lines.append(f"### Top {min(r['K'], TOP_N_TABLE)} clusters by size")
        lines.append("")
        lines.append("| cluster id | size | NA frac | dominant overall | "
                      "physical purity | physical dominant |")
        lines.append("|------------|------|---------|------------------|"
                      "-----------------|-------------------|")
        for idx in sort_idx:
            cid = r["cluster_ids"][idx] if "cluster_ids" in r else idx
            sz = r["cluster_sizes"][idx]
            cm_row = r["confusion"][idx]
            dom = r["spoca_class_names"][int(np.argmax(cm_row))] if sz else "—"
            na_f = r.get("na_frac_per_cluster", [0]*r["K"])[idx]
            pp = r.get("physical_purity_per_cluster", [0]*r["K"])[idx]
            phys_dom_id = r.get(
                "physical_dominant_class_per_cluster", [None]*r["K"]
            )[idx]
            phys_dom_name = (
                "—" if phys_dom_id is None
                else MASK_CLASS_NAMES.get(phys_dom_id, str(phys_dom_id))
            )
            lines.append(f"| {cid} | {sz:,} | {na_f:.3f} | {dom} "
                          f"| {pp:.3f} | {phys_dom_name} |")
        if r["K"] > TOP_N_TABLE:
            lines.append(f"| ... | ... | ... | ... | ... | "
                          f"(K - {TOP_N_TABLE}) more |")
        lines.append("")

        # Top non-NA-dominant clusters — most relevant for on-disk findings.
        nna_idx = [i for i in range(r["K"])
                    if r.get("dominant_class_per_cluster", [])[i] != -1]
        if nna_idx:
            nna_idx_sorted = sorted(nna_idx,
                key=lambda i: -r["cluster_sizes"][i])[:TOP_N_TABLE]
            lines.append(f"### Top non-NA-dominant clusters (by size)")
            lines.append("")
            lines.append("| cluster id | size | NA frac | dominant | "
                          "purity | physical purity |")
            lines.append("|------------|------|---------|----------|"
                          "--------|-----------------|")
            for idx in nna_idx_sorted:
                cid = r["cluster_ids"][idx] if "cluster_ids" in r else idx
                sz  = r["cluster_sizes"][idx]
                naf = r["na_frac_per_cluster"][idx]
                pu  = r["purity_per_cluster"][idx]
                pp  = r["physical_purity_per_cluster"][idx]
                cm_row = r["confusion"][idx]
                dom = r["spoca_class_names"][int(np.argmax(cm_row))]
                lines.append(f"| {cid} | {sz:,} | {naf:.3f} | {dom} "
                              f"| {pu:.3f} | {pp:.3f} |")
            if len(nna_idx) > TOP_N_TABLE:
                lines.append(f"| ... | ... | ... | ... | ... | "
                              f"({len(nna_idx) - TOP_N_TABLE}) more |")
            lines.append("")

        # Hidden-physical candidates — the SPoCA-mislabel candidates.
        n_hidden = int(r.get("n_hidden_physical_candidates", 0))
        if n_hidden > 0:
            HIDDEN_TABLE_LIMIT = 30
            # Need to load the detail file's ordering (sorted by physical
            # count desc).  Just sort here from results.
            cand_idx = [i for i in range(r["K"])
                         if r["dominant_class_per_cluster"][i] == -1
                         and r["physical_purity_per_cluster"][i]
                              >= r.get("hidden_purity_threshold", 0.7)
                         and (1 - r["na_frac_per_cluster"][i])
                              >= r.get("hidden_physical_frac_min", 0.05)]
            cand_idx_sorted = sorted(
                cand_idx,
                key=lambda i: -r["physical_count_per_cluster"][i],
            )[:HIDDEN_TABLE_LIMIT]
            lines.append(f"### Hidden-physical candidates "
                          f"(NA-dominant overall but ≥"
                          f"{int(100*r.get('hidden_purity_threshold', 0.7))}% "
                          f"physical purity)")
            lines.append("")
            lines.append("These clusters are dominated by NA in their overall "
                          "label distribution, but their non-NA members agree "
                          "strongly on a single on-disk class.  They are the "
                          "leading candidates for SPoCA threshold misses — "
                          "patches the heuristic labeled NA but the model "
                          "represents as on-disk structure (limb features, "
                          "prominences, jets, etc.).")
            lines.append("")
            lines.append(f"- Total candidates: **{n_hidden}**")
            lines.append(f"- Detailed list (sorted by physical-member count): "
                          f"`cluster_purity/cluster_purity_hidden_physical.json`")
            lines.append("")
            lines.append("| cluster id | size | NA frac | physical "
                          "members | physical purity | physical dominant |")
            lines.append("|------------|------|---------|-------------------"
                          "|-----------------|-------------------|")
            for idx in cand_idx_sorted:
                cid = r["cluster_ids"][idx] if "cluster_ids" in r else idx
                sz  = r["cluster_sizes"][idx]
                naf = r["na_frac_per_cluster"][idx]
                pc  = r["physical_count_per_cluster"][idx]
                pp  = r["physical_purity_per_cluster"][idx]
                phys_dom_id = r["physical_dominant_class_per_cluster"][idx]
                phys_dom_name = (
                    "—" if phys_dom_id is None
                    else MASK_CLASS_NAMES.get(phys_dom_id, str(phys_dom_id))
                )
                lines.append(f"| {cid} | {sz:,} | {naf:.3f} | {pc:,} "
                              f"| {pp:.3f} | {phys_dom_name} |")
            if len(cand_idx) > HIDDEN_TABLE_LIMIT:
                lines.append(f"| ... | ... | ... | ... | ... | "
                              f"({len(cand_idx) - HIDDEN_TABLE_LIMIT}) more |")
            lines.append("")

        lines.append(f"![umap by mask class](umap_cluster/umap_by_mask_class.png)")
        lines.append(f"![umap by cluster](umap_cluster/umap_by_cluster.png)")
        lines.append(f"![umap by position](umap_cluster/umap_by_position.png)")
        lines.append(f"![cluster purity](cluster_purity/cluster_purity.png)")
        lines.append(f"![cluster atlas — NA frac × physical purity]"
                      f"(cluster_purity/cluster_atlas_na_vs_physical_purity.png)")
        if r["K"] >= 5:
            lines.append(f"![cluster size distribution]"
                          f"(cluster_purity/cluster_size_distribution.png)")
        lines.append("")

    if "hidden_physical_spatial" in results:
        h = results["hidden_physical_spatial"]
        lines.append("## Hidden-physical candidates — spatial distribution")
        lines.append("")
        n_hidden = int(h.get("n_hidden_clusters", 0))
        if n_hidden == 0:
            lines.append("No hidden-physical candidates were found "
                          "(or `--exclude-na-from-clustering` was on).")
            lines.append("")
        else:
            lines.append(f"- {n_hidden} hidden-physical clusters, "
                          f"{int(h.get('total_hidden_patches', 0)):,} total "
                          f"patches across all samples")
            lines.append("")
            lines.append("### Hidden-physical patch counts by physical class")
            lines.append("")
            lines.append("| physical class | n clusters | n patches |")
            lines.append("|----------------|-----------:|----------:|")
            by_cls = h.get("by_physical_class", {})
            for cls_name in ("Coronal Hole", "Quiet Sun", "Active Region"):
                if cls_name in by_cls:
                    s = by_cls[cls_name]
                    lines.append(
                        f"| {cls_name} | {s.get('n_clusters', 0)} "
                        f"| {s.get('n_hidden_patches', 0):,} |"
                    )
            lines.append("")
            lines.append(f"![hidden-physical density]"
                          f"(hidden_physical_spatial/hidden_physical_density.png)")
            lines.append(f"![hidden-physical density per class]"
                          f"(hidden_physical_spatial/hidden_physical_density_per_class.png)")
            lines.append(f"![hidden-physical per-sample maps]"
                          f"(hidden_physical_spatial/hidden_physical_per_sample.png)")
            # Single-sample close-up of the last sample
            last_idx = h.get("last_sample_index")
            last_ts  = h.get("last_sample_timestamp")
            last_counts = h.get("last_sample_hidden_counts", {})
            if last_idx is not None:
                ts_str = f", timestamp = {last_ts}" if last_ts else ""
                lines.append("")
                lines.append(f"### Last sample close-up "
                              f"(sample {last_idx}{ts_str})")
                lines.append("")
                if last_counts:
                    cnt_bits = [
                        f"{cls_name}: {n:,}" for cls_name, n in last_counts.items()
                        if n > 0
                    ]
                    if cnt_bits:
                        lines.append(f"Hidden-physical tile counts in this "
                                      f"sample: {', '.join(cnt_bits)}.")
                lines.append("")
                lines.append(f"![hidden-physical last sample]"
                              f"(hidden_physical_spatial/hidden_physical_last_sample.png)")
            lines.append("")

    if "pca_viz" in results:
        pv = results["pca_viz"]
        lines.append("## PCA RGB visualization")
        lines.append("")
        space_mode = pv.get("pca_viz_on", "residual")
        norm_mode  = pv.get("pca_viz_norm", "per_sample")
        lines.append(f"- Embedding space(s): `{space_mode}` "
                      f"(raw shows position+content; residual shows content-only)")
        lines.append(f"- Color normalization: `{norm_mode}` "
                      f"(per-sample = within-sample contrast; "
                      f"global = cross-sample comparable)")
        if "explained_variance_ratio_3_raw" in pv:
            ev = pv["explained_variance_ratio_3_raw"]
            lines.append(f"- 3-component EV (raw): "
                          f"{sum(ev):.3f}  "
                          f"(per-PC: {ev[0]:.3f}, {ev[1]:.3f}, {ev[2]:.3f})")
        if "explained_variance_ratio_3_residual" in pv:
            ev = pv["explained_variance_ratio_3_residual"]
            lines.append(f"- 3-component EV (residual): "
                          f"{sum(ev):.3f}  "
                          f"(per-PC: {ev[0]:.3f}, {ev[1]:.3f}, {ev[2]:.3f})")
        if ("explained_variance_ratio_3_raw" in pv
                and "explained_variance_ratio_3_residual" in pv):
            ev_r = sum(pv["explained_variance_ratio_3_raw"])
            ev_s = sum(pv["explained_variance_ratio_3_residual"])
            lines.append(f"- Δ (raw − residual): {ev_r - ev_s:+.3f}  "
                          f"— larger Δ means position contributes more "
                          f"low-rank variance")
        n_rendered = pv["n_samples_rendered"]
        lines.append(f"- {n_rendered} sample panels in `pca_viz/`")
        lines.append("")
        lines.append("Per-patch RGB = first 3 PCs of the embedding space "
                      "(raw or residual).  When norm = `per_sample`, each "
                      "component is normalized using within-sample 2nd-98th "
                      "percentiles — colors comparable WITHIN a sample but "
                      "not across samples.  When norm = `global`, "
                      "percentiles are computed across all samples — colors "
                      "are cross-sample comparable.")
        lines.append("")

    md_path.write_text("\n".join(lines))
    print(f"\n[report] Wrote {md_path}")


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = args.ablation_dir / "unified_probing"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.out_dir}")
    print(f"Probes: {args.probes}")

    # 1. Load ablation outputs
    ab = load_ablation_outputs(args.ablation_dir)
    if ab.N < 2:
        raise RuntimeError("Need at least 2 samples for any probe.")

    # 2. Obtain raw embeddings
    if args.reextract_raw:
        raw = reextract_raw_embeddings(ab)
    else:
        raw = reconstruct_raw_from_residuals(ab)
        # Sanity check: raw - mu[keep_idx] should equal residuals exactly.
        # Done per-sample to avoid a 203 GB intermediate temporary.
        mu_kept = ab.mu[ab.keep_idx]  # (n_keep, D), already in RAM
        diff = 0.0
        for _t in range(ab.N):
            diff = max(diff, float(np.abs(raw[_t] - mu_kept - ab.residuals[_t]).max()))
        print(f"      reconstruction max abs residual: {diff:.2e}")
        if diff > 1e-3:
            warnings.warn(f"Raw reconstruction mismatch ({diff:.2e}); "
                           f"consider --reextract-raw for verification")

    # 3. Pixel targets (only needed for two probes)
    pixels = None
    pixel_dependent = {"linear_probe_px", "linear_probe_px_stratified", "pca_viz"}
    if any(p in pixel_dependent for p in args.probes):
        pixels = load_patch_pixel_targets(ab)

    # 3b. Optional PCA reduction for the linear probes only.  Other probes
    # keep operating on full-D embeddings so their numbers stay comparable
    # across runs and across the literature.
    if (args.use_pca_for_linear_probe and
            ("linear_probe_px" in args.probes or
             "linear_probe_cls" in args.probes)):
        raw_lp, res_lp = _pca_reduce_for_probe(raw, ab.residuals, ab, args)
    else:
        raw_lp, res_lp = raw, ab.residuals

    # 4. Run probes
    results = {}

    if "effective_rank" in args.probes:
        results["effective_rank"] = probe_effective_rank(
            raw, ab.residuals, ab.mask_labels, args,
            args.out_dir / "effective_rank")

    if "pca_ev" in args.probes:
        results["pca_ev"] = probe_pca_ev(
            raw, ab.residuals, ab.mask_labels, args,
            args.out_dir / "pca_ev")

    if "linear_probe_px" in args.probes:
        results["linear_probe_px"] = probe_linear_probe_pixels(
            raw_lp, res_lp, pixels, args,
            args.out_dir / "linear_probe_px")

    if "linear_probe_px_stratified" in args.probes:
        results["linear_probe_px_stratified"] = probe_linear_probe_pixels_stratified(
            raw_lp, res_lp, pixels, ab.mask_labels, args,
            args.out_dir / "linear_probe_px_stratified")

    if "linear_probe_cls" in args.probes:
        results["linear_probe_cls"] = probe_linear_probe_class(
            raw_lp, res_lp, ab.mask_labels, args,
            args.out_dir / "linear_probe_cls")

    if "spatial_corr" in args.probes:
        results["spatial_corr"] = probe_spatial_corr(
            raw, ab.residuals, ab.positions, args,
            args.out_dir / "spatial_corr")

    cluster_labels = None
    if "umap_cluster" in args.probes:
        umap_res = probe_umap_cluster(
            ab, args, args.out_dir / "umap_cluster")
        cluster_labels = umap_res["cluster_labels"]
        results["umap_cluster"] = {"method": umap_res["method"],
                                    "k":      umap_res["k"],
                                    "cluster_on": umap_res.get("cluster_on", "?"),
                                    "cluster_space_tag":
                                        umap_res.get("cluster_space_tag", "?"),
                                    "exclude_na": umap_res.get("exclude_na", False),
                                    "n_clustered": umap_res.get("n_clustered", 0),
                                    "n_excluded":  umap_res.get("n_excluded", 0),
                                    "info":   umap_res.get("info", {})}

    if "cluster_purity" in args.probes:
        if cluster_labels is None:
            # Try to load from a previous run
            try:
                cluster_labels = np.load(
                    args.out_dir / "umap_cluster" / "cluster_labels.npy")
            except FileNotFoundError:
                print("[probe 7/8] cluster_purity skipped — run umap_cluster first")
        if cluster_labels is not None:
            results["cluster_purity"] = probe_cluster_purity(
                cluster_labels, ab.mask_labels, args,
                args.out_dir / "cluster_purity")

    if "hidden_physical_spatial" in args.probes:
        if cluster_labels is None:
            # Try to load from a previous run (umap_cluster output)
            try:
                cluster_labels = np.load(
                    args.out_dir / "umap_cluster" / "cluster_labels.npy")
            except FileNotFoundError:
                print("[probe 7b] hidden_physical_spatial skipped — "
                      "no cluster_labels available (run umap_cluster first)")
        cp_results = results.get("cluster_purity")
        if cp_results is None:
            # Try to load cluster_purity.json from a previous run
            try:
                cp_path = args.out_dir / "cluster_purity" / "cluster_purity.json"
                cp_results = json.loads(cp_path.read_text())
            except FileNotFoundError:
                print("[probe 7b] hidden_physical_spatial skipped — "
                      "no cluster_purity results available (run cluster_purity "
                      "first)")
                cp_results = None
        if cluster_labels is not None and cp_results is not None:
            results["hidden_physical_spatial"] = probe_hidden_physical_spatial(
                cluster_labels, ab, cp_results, args,
                args.out_dir / "hidden_physical_spatial")

    if "pca_viz" in args.probes:
        results["pca_viz"] = probe_pca_viz(
            raw, ab.residuals, pixels, ab, args,
            args.out_dir / "pca_viz")

    # 5. Report
    write_unified_report(results, ab, args, args.out_dir)
    print(f"\nDone. Top-level report: {args.out_dir / 'unified_report.md'}")


if __name__ == "__main__":
    main()