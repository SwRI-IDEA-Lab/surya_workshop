#!/usr/bin/env python3
"""
Per-position mean-ablation harness for Surya patch embeddings.

Motivation
----------
Diagnostic UMAP of Surya's deepest backbone output shows a quadrupolar
2-torus structure driven by Fourier position encoding leaking through the
residual stream of the SpectFormer backbone (lowest-frequency components
of the position basis are {sin 2πx, cos 2πx, sin 2πy, cos 2πy}).  This
script removes the position-conditional component of each token embedding
so that downstream UMAP / HDBSCAN can surface content-driven structure.

Method
------
For each spatial token position i ∈ {0, …, 65535} we estimate

    μ[i] := E_t[ X[t, i, :] ]      (mean over time samples)

and replace each embedding by its residual

    R[t, i, :] := X[t, i, :] − μ[i].

The decomposition

    Σ_{t,i} ‖X[t,i] − μ_global‖² = N · Σ_i ‖μ[i] − μ_global‖²
                                     + Σ_{t,i} ‖X[t,i] − μ[i]‖²

splits the total (centered) variance into between-position and
within-position components.  The "between-position fraction" is the
quantity to look at: large (≳0.5) means position dominates.

Memory model
------------
For 200 samples × 65,536 patches × 1,280 dims × 2 B (float16) ≈ 34 GB on
disk via numpy memmap, never held entirely in RAM.  Storage dtype is
configurable.  The per-position sum is accumulated in float64 to avoid
precision drift across many samples.

Outputs in --out-dir
--------------------
  extraction_index.csv          The samples used.
  patch_pos_mean.npy            (N_spatial, D)  per-position mean (float32).
  patch_residuals.npy           (N, n_keep, D)  ablated embeddings (float32).
  patch_residuals_positions.npy (n_keep, 2)     [row, col] of kept patches.
  patch_mask_labels.npy         (N, n_keep)     mask class per kept patch.
  timestamps.npy                (N,)
  variance_accounting.txt       between/within-position variance breakdown.
  spatial_pos_mean_norm.png     spatial map of ‖μ[i]‖ per (i_row, i_col).
  patch_residuals_pca.npy       (N, n_keep, k) PCA-projected residuals.
  pca_explained_variance_ratio.npy   (k,) variance ratio per PC.
  pca_components.npy            (k, D) PCA basis in embedding space.
  pca_scree.png                 cumulative explained-variance scree plot.

The ``patch_residuals.npy`` array is shape-compatible with the existing
``patch_embeddings.npy`` produced by ``embedding_analysis.py``; you can
swap one for the other in the existing UMAP/HDBSCAN/plotting code.

Assumptions worth flagging
--------------------------
* Per-position mean ablation removes *anything* that is deterministic
  given pixel position.  That includes the explicit Fourier position
  encoding **and** real per-pixel climatology (limb darkening, on/off-disk
  geometry, the typical latitudinal distribution of active regions, etc.).
  For cluster *discovery* this is what we want; for occurrence statistics
  of features at their typical locations, residuals will undercount.
* Pixel position ≠ heliographic position.  The Sun rotates beneath the
  pixel grid; ablation works in detector coordinates, which is the right
  frame for removing the position-encoding artifact but not for removing
  feature-class climatology in heliographic coordinates.
* Stable per-position means require many samples (rule of thumb: ≥100,
  comfortable: ≥300).  Below ~30 samples the means are noisy enough that
  ablation injects new noise at each position.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# --- locate repo root, identical convention to embedding_analysis.py ---------
_script_dir = Path(__file__).resolve().parent
_repo_root  = _script_dir
while not (_repo_root / "workshop_infrastructure").exists() and _repo_root != _repo_root.parent:
    _repo_root = _repo_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Import shared configuration and helpers from the existing analysis script.
from embedding_analysis import (
    INDEX_PATH, MASKS_DIR, MASKS_TIME_TOLERANCE,
    SCALERS_PATH, CHECKPOINT_PATH, CACHE_PATH,
    CHANNELS, MODEL_CONFIG,
    PATCH_SIZE, PATCH_GRID,
    DEVICE,
    subsample_index, load_mask_timestamps, intersect_with_masks,
    extract_patch_labels,
    build_backbone,
)
from workshop_infrastructure.datasets.helio import HelioNetCDFDataset
from workshop_infrastructure.utils import build_scalers


# === defaults =================================================================
DEFAULT_N_SAMPLES     = 200      # rule of thumb; below ~30 ablation gets noisy
DEFAULT_KEEP_PATCHES  = 8192     # for the downstream residuals .npy
DEFAULT_OUT_DIR       = _script_dir / "ablation_outputs"
DEFAULT_STORAGE_DTYPE = "float16"  # halves on-disk size vs float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--start-date",   default="2013-01-01")
    p.add_argument("--end-date",     default="2020-12-31")
    p.add_argument("--n-samples",    type=int, default=DEFAULT_N_SAMPLES,
                   help="Number of timestamps drawn uniformly from [start, end].")
    p.add_argument("--keep-patches", type=int, default=DEFAULT_KEEP_PATCHES,
                   help="Patches kept per sample in the residuals .npy.")
    p.add_argument("--out-dir",      type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--storage-dtype", default=DEFAULT_STORAGE_DTYPE,
                   choices=["float16", "float32"],
                   help="Dtype for the on-disk memmap of raw full-grid embeddings.")
    p.add_argument("--keep-raw-memmap", action="store_true",
                   help="Keep the raw embeddings memmap on disk after running.")
    p.add_argument("--pca-components", type=int, default=50,
                   help="If > 0, run PCA on residuals and save a reduced-dim "
                        "version (recommended UMAP input). Set to 0 to disable. "
                        "Default 50 components covers ≳90%% of residual variance "
                        "in typical ViT-style backbones.")
    p.add_argument("--triage", action="store_true",
                   help="Variance-accounting only: run Pass 1 (extraction) and "
                        "compute the per-position mean + variance breakdown, then "
                        "exit without writing residuals or mask labels.  Pair with "
                        "small --n-samples (e.g. 30) for a fast diagnostic before "
                        "committing to a full extraction.")
    return p.parse_args()


def variance_accounting(raw_memmap, mu, N):
    """
    Compute the centered variance decomposition:

        Σ ||X[t,i] - μ_global||² = N · Σ ||μ[i] - μ_global||² + Σ ||X[t,i] - μ[i]||²
        ─────────────────────────   ──────────────────────────   ──────────────────────
              total centered             between-position            within-position
                                        (position-conditional)         (residual)

    Returns (total, between, within), all float64.
    """
    mu_f32     = mu.astype(np.float32)
    mu_global  = mu_f32.mean(axis=0)                                  # (D,)
    between_sq = float(N * ((mu.astype(np.float64) - mu_global.astype(np.float64)) ** 2).sum())

    total_sq  = 0.0
    within_sq = 0.0
    for t in tqdm(range(N), desc="Variance accounting", leave=False):
        x = raw_memmap[t].astype(np.float32)
        total_sq  += float(((x - mu_global) ** 2).sum())
        within_sq += float(((x - mu_f32) ** 2).sum())
    return total_sq, between_sq, within_sq


def plot_pos_mean_spatial(mu: np.ndarray, patch_grid: int, out_path: Path) -> None:
    """Spatial heatmap of ‖μ[i]‖ across the patch grid — diagnostic only."""
    norms = np.linalg.norm(mu, axis=-1).reshape(patch_grid, patch_grid)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        norms, origin="upper", cmap="inferno",
        vmin=np.nanpercentile(norms, 2),
        vmax=np.nanpercentile(norms, 98),
    )
    plt.colorbar(im, ax=ax).set_label(r"$\|\mu[i]\|$  (per-position mean)")
    ax.set_xlabel("Patch column")
    ax.set_ylabel("Patch row")
    ax.set_title("Per-position embedding mean magnitude")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def plot_pca_scree(explained_variance_ratio: np.ndarray, out_path: Path) -> None:
    """Cumulative explained variance plot with threshold annotations."""
    cumvar = np.cumsum(explained_variance_ratio)
    n = len(cumvar)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, n + 1), cumvar, "o-", markersize=4, linewidth=1.5)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_title("PCA scree (post-ablation residuals)")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.02)

    for thresh in (0.5, 0.9, 0.95):
        if cumvar[-1] >= thresh:
            k = int(np.searchsorted(cumvar, thresh)) + 1
            ax.axhline(thresh, color="gray", linestyle=":", alpha=0.4)
            ax.annotate(
                f"{int(thresh*100)}%: {k} comp",
                xy=(k, thresh),
                xytext=(k + max(2, n * 0.02), thresh - 0.03),
                fontsize=9, alpha=0.8,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    storage_dtype = np.dtype(args.storage_dtype)

    print(f"Device : {DEVICE}")
    print(f"Output : {args.out_dir}")

    # --- 1. Index + masks ---------------------------------------------------
    print("\n[1/6] Loading index and intersecting with masks ...")
    index_df = pd.read_csv(INDEX_PATH)
    index_df["timestep"] = pd.to_datetime(index_df["timestep"])
    index_df = index_df[index_df["present"] == 1].reset_index(drop=True)
    mask_df  = load_mask_timestamps(MASKS_DIR)
    index_df = intersect_with_masks(index_df, mask_df, MASKS_TIME_TOLERANCE)
    embed_df = subsample_index(index_df, args.start_date, args.end_date, args.n_samples)

    embed_index_path = args.out_dir / "extraction_index.csv"
    embed_df.drop(columns=["mask_path"]).to_csv(embed_index_path, index=False)
    print(f"  {len(embed_df)} samples selected -> {embed_index_path}")

    # --- 2. Backbone + dataset ---------------------------------------------
    print("\n[2/6] Building backbone and dataset ...")
    backbone = build_backbone(CHECKPOINT_PATH, DEVICE)
    scalers  = build_scalers(SCALERS_PATH)
    dataset  = HelioNetCDFDataset(
        index_path                = str(embed_index_path),
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
        dataset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=DEVICE.type == "cuda",
    )
    timestamps = pd.to_datetime([str(ts) for ts in dataset.valid_indices])
    N_actual   = len(dataset)

    # Align mask paths to the timestamps actually loaded.
    ts_to_mask = dict(zip(embed_df["timestep"], embed_df["mask_path"]))
    mask_paths = [ts_to_mask[ts] for ts in timestamps]
    print(f"  {N_actual} samples  ({timestamps[0].date()} → {timestamps[-1].date()})")

    # --- 3. Pass 1: extract full-grid embeddings -> memmap, accumulate sum --
    N_spatial = PATCH_GRID * PATCH_GRID
    D         = MODEL_CONFIG["embed_dim"]

    raw_path = args.out_dir / f"raw_embeddings_{storage_dtype.name}.dat"
    raw = np.memmap(
        raw_path, dtype=storage_dtype, mode="w+",
        shape=(N_actual, N_spatial, D),
    )
    raw_size_gb = raw.nbytes / 1e9

    print(f"\n[3/6] Pass 1: extracting full-grid embeddings for {N_actual} samples")
    print(f"      memmap : {raw_path}  ({raw_size_gb:.1f} GB on disk, {storage_dtype})")

    running_sum = np.zeros((N_spatial, D), dtype=np.float64)
    backbone.eval()
    with torch.no_grad():
        for t, batch in enumerate(tqdm(loader, total=N_actual, desc="Extracting")):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            tokens = backbone(batch)                  # (1, N_spatial, D), finetune=True
            arr32  = tokens[0].cpu().float().numpy()  # (N_spatial, D), float32
            running_sum += arr32.astype(np.float64)
            raw[t]      = arr32.astype(storage_dtype)
    raw.flush()

    mu = (running_sum / N_actual).astype(np.float32)  # (N_spatial, D)
    np.save(args.out_dir / "patch_pos_mean.npy", mu)
    print(f"  saved per-position mean -> patch_pos_mean.npy")

    # --- 4. Variance accounting --------------------------------------------
    print("\n[4/6] Variance accounting ...")
    total_sq, between_sq, within_sq = variance_accounting(raw, mu, N_actual)
    frac_between = between_sq / total_sq if total_sq > 0 else float("nan")
    frac_within  = within_sq  / total_sq if total_sq > 0 else float("nan")

    report = (
        f"Surya embedding ablation — variance accounting\n"
        f"=================================================\n"
        f"N_samples                 : {N_actual}\n"
        f"N_spatial                 : {N_spatial}\n"
        f"D                         : {D}\n"
        f"\n"
        f"Total centered variance   : {total_sq:.4e}\n"
        f"Between-position variance : {between_sq:.4e}   "
        f"({frac_between:.3f} of total)\n"
        f"Within-position variance  : {within_sq:.4e}   "
        f"({frac_within:.3f} of total)\n"
        f"\n"
        f"Interpretation:\n"
        f"  Between-position fraction is the share of variance that is\n"
        f"  deterministic given pixel position (= position encoding +\n"
        f"  per-pixel climatology).  Values ≳ 0.5 indicate position\n"
        f"  dominates the embedding and ablation is essential before\n"
        f"  content-clustering analysis.\n"
    )
    print(report)
    (args.out_dir / "variance_accounting.txt").write_text(report)
    plot_pos_mean_spatial(mu, PATCH_GRID, args.out_dir / "spatial_pos_mean_norm.png")

    # --- Triage exit --------------------------------------------------------
    if args.triage:
        print("\n[triage] Skipping Pass 2 (residuals + mask labels).")
        print(f"[triage] Read {args.out_dir/'variance_accounting.txt'} for the diagnostic.")
        print(f"[triage] If between-position fraction is large (≳0.5), position dominates")
        print(f"[triage] and a full ablation run (drop --triage, raise --n-samples) is warranted.")

        # Memmap cleanup also runs in triage mode.
        if not args.keep_raw_memmap:
            del raw
            try:
                raw_path.unlink()
                print(f"[triage] Removed raw memmap ({raw_path}).")
            except OSError as e:
                print(f"[triage] Couldn't remove memmap ({raw_path}): {e}")
        else:
            print(f"[triage] Raw memmap retained at {raw_path}")
        return

    # --- 5. Pass 2: subsample positions, compute residuals ------------------
    print(f"\n[5/6] Pass 2: subsampling {args.keep_patches} patches and computing residuals ...")
    keep_idx = np.sort(rng.choice(
        N_spatial, size=min(args.keep_patches, N_spatial), replace=False,
    ))
    rows   = (keep_idx // PATCH_GRID).astype(np.int32)
    cols   = (keep_idx  % PATCH_GRID).astype(np.int32)
    n_keep = len(keep_idx)

    residuals = np.empty((N_actual, n_keep, D), dtype=np.float32)
    mu_kept   = mu[keep_idx]                          # (n_keep, D)
    for t in tqdm(range(N_actual), desc="Residuals"):
        residuals[t] = raw[t, keep_idx, :].astype(np.float32) - mu_kept

    np.save(args.out_dir / "patch_residuals.npy", residuals)
    np.save(args.out_dir / "patch_residuals_positions.npy",
            np.stack([rows, cols], axis=1))
    np.save(args.out_dir / "timestamps.npy", timestamps.values)

    # Mask labels using the existing helper (only kept positions).
    print("  Extracting mask labels for kept patches ...")
    patch_labels = extract_patch_labels(mask_paths, keep_idx, PATCH_GRID, PATCH_SIZE)
    np.save(args.out_dir / "patch_mask_labels.npy", patch_labels)

    # --- 6. Optional PCA preprocessing for downstream UMAP -----------------
    if args.pca_components > 0:
        print(f"\n[6/6] PCA preprocessing → {args.pca_components} components ...")
        from sklearn.decomposition import PCA

        # Reshape (N, n_keep, D) → (N*n_keep, D); reshape is a view, no copy.
        flat = residuals.reshape(-1, D)
        pca  = PCA(
            n_components = args.pca_components,
            svd_solver   = "randomized",
            random_state = args.seed,
        )
        flat_pca      = pca.fit_transform(flat).astype(np.float32)
        residuals_pca = flat_pca.reshape(N_actual, n_keep, args.pca_components)

        np.save(args.out_dir / "patch_residuals_pca.npy", residuals_pca)
        np.save(args.out_dir / "pca_explained_variance_ratio.npy",
                pca.explained_variance_ratio_.astype(np.float32))
        np.save(args.out_dir / "pca_components.npy",
                pca.components_.astype(np.float32))
        plot_pca_scree(pca.explained_variance_ratio_,
                       args.out_dir / "pca_scree.png")

        cumvar = np.cumsum(pca.explained_variance_ratio_)
        print(f"  Top {args.pca_components} PCs cumulative variance: "
              f"{cumvar[-1]*100:.2f}% of post-ablation residual")
        for thresh in (0.5, 0.9, 0.95, 0.99):
            if cumvar[-1] >= thresh:
                k = int(np.searchsorted(cumvar, thresh)) + 1
                print(f"  {int(thresh*100):2d}% reached at {k} components")
            else:
                print(f"  {int(thresh*100):2d}% not reached within "
                      f"{args.pca_components} components")
    else:
        print("\n[6/6] PCA skipped (--pca-components 0)")

    # --- Cleanup ------------------------------------------------------------
    if not args.keep_raw_memmap:
        del raw  # release handle before unlinking
        try:
            raw_path.unlink()
            print(f"\nRemoved raw memmap ({raw_path}). Use --keep-raw-memmap to keep it.")
        except OSError as e:
            print(f"\nCouldn't remove memmap ({raw_path}): {e}")
    else:
        print(f"\nRaw memmap retained at {raw_path}")

    print("\nDone.")
    print(f"Drop-in for the existing UMAP/HDBSCAN pipeline:")
    if args.pca_components > 0:
        print(f"  RECOMMENDED: feed {args.out_dir/'patch_residuals_pca.npy'} to UMAP")
        print(f"               (shape (N, n_keep, {args.pca_components}), much faster)")
        print(f"  Raw residuals also available at {args.out_dir/'patch_residuals.npy'}")
    else:
        print(f"  use {args.out_dir/'patch_residuals.npy'} in place of patch_embeddings.npy")
    print(f"  use {args.out_dir/'patch_residuals_positions.npy'} in place of patch_positions")
    print(f"  use {args.out_dir/'patch_mask_labels.npy'} in place of patch_mask_labels.npy")


if __name__ == "__main__":
    main()
