#!/usr/bin/env python3
"""
Spatial visualization of top PCA components from the post-ablation residuals.

Each PC is a direction in 1280-d embedding space.  Projecting the residual at
patch position (i, j) for sample t onto PC_k yields a scalar
proj[t, i, j, k].  Plotting that scalar as a 256×256 spatial map shows what
the PC "is" — whether it lights up active regions, traces coronal hole
boundaries, picks out the limb, etc.

This script produces three families of figures:

1. pc_time_average.png
   For each top PC, the spatial map of the time-averaged projection
   (mean across all samples).  Time-averaging emphasizes structure that
   is consistent across samples — which after position ablation can
   indicate either residual position-conditional structure (covariance,
   not just mean) or persistent climatology.

2. pc_by_mask_class.png
   Bar chart per PC: mean PC projection within each mask class
   (Active Region / Quiet Sun / Coronal Hole / NA).  Direct test of
   whether the PC discriminates between physical labels.  PC sign is
   arbitrary; what matters is the contrast between classes.

3. pcNN_per_sample.png  (one figure per top --detail-pcs PCs)
   Spatial map of the PC projection for each sample.  Shows whether the
   PC tracks instantaneous solar features that move/evolve — and
   complements (1) by exposing per-sample variability that gets averaged
   out in the time-mean.

Outputs in --out-dir:
  pc_time_average.png
  pc_by_mask_class.png
  pcNN_per_sample.png  (NN = 01 .. detail_pcs)
  pc_projections.npy   (N, n_keep, top_k)  — saved for downstream analysis.

Assumptions worth flagging
--------------------------
* PC sign is arbitrary.  PC_k and -PC_k carry identical information.  So
  an "AR is positive, QS is negative" reading is the same as the reverse.
* PCA finds high-variance directions, not high-content-signal directions.
  A noisy dimension that varies a lot will get a high PC rank even if it
  contains no physical signal.  The mask-class bar chart is the primary
  check that a PC is content-meaningful: if the means are
  indistinguishable between classes, the PC is likely tracking
  noise or position-conditional covariance rather than physical structure.
* Time-averaging assumes solar features are uncorrelated across samples
  in pixel coordinates.  At ~3-month spacing this holds (rotation
  decorrelates).  At dense temporal spacing it would not.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- locate repo root, identical convention to embedding_analysis.py ---------
_script_dir = Path(__file__).resolve().parent
_repo_root  = _script_dir
while not (_repo_root / "workshop_infrastructure").exists() and _repo_root != _repo_root.parent:
    _repo_root = _repo_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from embedding_analysis import (
    PATCH_GRID,
    MASK_CLASS_NAMES,
    MASK_PALETTE,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ablation-dir", type=Path, required=True,
                   help="Directory produced by embedding_ablation.py "
                        "(must contain pca_components.npy etc.).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: <ablation-dir>/pca_spatial/).")
    p.add_argument("--top-k", type=int, default=10,
                   help="Number of top PCs to visualize in summary figures.")
    p.add_argument("--detail-pcs", type=int, default=4,
                   help="Number of top PCs to give detailed per-sample figures for.")
    p.add_argument("--clip-percentile", type=float, default=98.0,
                   help="Symmetric color clip percentile for diverging maps.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def project_onto_pcs(residuals: np.ndarray, components: np.ndarray) -> np.ndarray:
    """
    Project residuals onto a set of PCs.

    residuals  : (N, n_keep, D)
    components : (K, D)            — rows are the PC basis vectors.
    returns    : (N, n_keep, K)    — proj[t, i, k] = ⟨X[t, i], pc_k⟩
    """
    return np.einsum("nkd,cd->nkc", residuals, components)


def projections_to_grid(projections: np.ndarray, positions: np.ndarray,
                         patch_grid: int) -> np.ndarray:
    """
    Scatter a flat (n_keep,) projection vector into a 2D grid.

    Unsampled positions are NaN.  With full-grid extraction every position
    is present and there are no NaNs.
    """
    rows = positions[:, 0].astype(np.int64)
    cols = positions[:, 1].astype(np.int64)
    grid = np.full((patch_grid, patch_grid), np.nan, dtype=np.float32)
    grid[rows, cols] = projections
    return grid


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_pc_time_average(grids: list, ev_ratios: np.ndarray, top_k: int,
                          clip_pct: float, out_path: Path) -> None:
    """One panel per PC, showing time-averaged spatial pattern."""
    n_cols = min(top_k, 5)
    n_rows = (top_k + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows))
    axes = np.atleast_2d(axes)

    for k in range(top_k):
        ax = axes[k // n_cols, k % n_cols]
        v = float(np.nanpercentile(np.abs(grids[k]), clip_pct))
        im = ax.imshow(grids[k], origin="upper", cmap="RdBu_r", vmin=-v, vmax=v)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        ax.set_title(f"PC{k+1}  ({ev_ratios[k]*100:.2f}% var)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    for k in range(top_k, n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")

    fig.suptitle("Time-averaged PC projection of residuals", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def plot_pc_by_mask_class(projections: np.ndarray, labels: np.ndarray,
                           ev_ratios: np.ndarray, top_k: int,
                           out_path: Path) -> None:
    """Bar chart per PC showing mean projection within each mask class."""
    K = projections.shape[-1]
    flat_proj   = projections.reshape(-1, K)
    flat_labels = labels.reshape(-1)
    classes     = sorted(MASK_CLASS_NAMES.keys())   # [-1, 0, 1, 2]

    n_cols = min(top_k, 5)
    n_rows = (top_k + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.8 * n_rows))
    axes = np.atleast_2d(axes)

    for k in range(top_k):
        ax = axes[k // n_cols, k % n_cols]
        means, sems, names, colors = [], [], [], []
        for c in classes:
            m = flat_labels == c
            if m.sum() == 0:
                continue
            x = flat_proj[m, k]
            means.append(x.mean())
            sems.append(x.std(ddof=1) / np.sqrt(m.sum()))
            names.append(MASK_CLASS_NAMES[c])
            colors.append(MASK_PALETTE[c])

        x = np.arange(len(names))
        ax.bar(x, means, yerr=sems, color=colors, capsize=3, edgecolor="k", linewidth=0.4)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"PC{k+1}  ({ev_ratios[k]*100:.2f}% var)", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)

    for k in range(top_k, n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")

    fig.suptitle("Mean PC projection by mask class (error bars = SEM)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def plot_pc_per_sample(projections: np.ndarray, positions: np.ndarray,
                        patch_grid: int, k: int, ev_ratio: float,
                        timestamps: np.ndarray, clip_pct: float,
                        out_path: Path) -> None:
    """Spatial map of PC k's projection at each sample timestamp."""
    proj_k = projections[:, :, k]                  # (N, n_keep)
    N      = proj_k.shape[0]
    v      = float(np.nanpercentile(np.abs(proj_k), clip_pct))

    n_cols = min(N, 4)
    n_rows = (N + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows))
    axes = np.atleast_2d(axes)
    last_im = None

    for t in range(N):
        ax = axes[t // n_cols, t % n_cols]
        grid = projections_to_grid(proj_k[t], positions, patch_grid)
        last_im = ax.imshow(grid, origin="upper", cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(str(timestamps[t])[:10], fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for t in range(N, n_rows * n_cols):
        axes[t // n_cols, t % n_cols].axis("off")

    fig.suptitle(f"PC{k+1} projection across {N} samples  "
                 f"({ev_ratio*100:.2f}% variance)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(),
                            fraction=0.025, pad=0.02)
        cbar.set_label("PC projection")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = args.ablation_dir / "pca_spatial"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.out_dir}")

    # --- 1. Load -----------------------------------------------------------
    print("\n[1/3] Loading residuals and PCA basis ...")
    residuals  = np.load(args.ablation_dir / "patch_residuals.npy")
    components = np.load(args.ablation_dir / "pca_components.npy")
    ev_ratios  = np.load(args.ablation_dir / "pca_explained_variance_ratio.npy")
    positions  = np.load(args.ablation_dir / "patch_residuals_positions.npy")
    labels     = np.load(args.ablation_dir / "patch_mask_labels.npy")
    timestamps = np.load(args.ablation_dir / "timestamps.npy")

    N, n_keep, D = residuals.shape
    K = components.shape[0]
    top_k    = min(args.top_k, K)
    detail_k = min(args.detail_pcs, top_k)

    print(f"  residuals  : {residuals.shape}")
    print(f"  components : {components.shape}  (using top {top_k})")
    print(f"  samples    : {N}")

    # --- 2. Project --------------------------------------------------------
    print(f"\n[2/3] Projecting residuals onto top {top_k} PCs ...")
    projections = project_onto_pcs(residuals, components[:top_k])  # (N, n_keep, top_k)
    np.save(args.out_dir / "pc_projections.npy", projections.astype(np.float32))
    print(f"  projections shape: {projections.shape}")

    # --- 3. Plots ----------------------------------------------------------
    print("\n[3/3] Generating figures ...")

    # 3a. Time-averaged spatial map per PC
    time_avg = projections.mean(axis=0)                          # (n_keep, top_k)
    grids = [
        projections_to_grid(time_avg[:, k], positions, PATCH_GRID)
        for k in range(top_k)
    ]
    plot_pc_time_average(
        grids, ev_ratios, top_k,
        args.clip_percentile,
        args.out_dir / "pc_time_average.png",
    )

    # 3b. Bar chart by mask class
    plot_pc_by_mask_class(
        projections, labels, ev_ratios, top_k,
        args.out_dir / "pc_by_mask_class.png",
    )

    # 3c. Per-sample detail figures for the top detail_k PCs
    for k in range(detail_k):
        plot_pc_per_sample(
            projections, positions, PATCH_GRID,
            k, float(ev_ratios[k]), timestamps,
            args.clip_percentile,
            args.out_dir / f"pc{k+1:02d}_per_sample.png",
        )

    print(f"\nDone. Outputs in {args.out_dir}/")
    print("\nInterpretation guide:")
    print("  pc_by_mask_class.png  — primary content-validity check.")
    print("                          Large between-class differences = PC is")
    print("                          discriminating physical features.")
    print("  pc_time_average.png   — time-averaged spatial structure.")
    print("                          Persistent disk-shaped pattern = real")
    print("                          climatology.  Banded patterns = position")
    print("                          covariance still leaking through.")
    print("  pcNN_per_sample.png   — instantaneous PC structure.  Should")
    print("                          track moving solar features over time.")


if __name__ == "__main__":
    main()
