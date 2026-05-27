#!/usr/bin/env python3
"""
precompute_pca.py — Compute PCA on patch residuals from embedding_ablation.py.

Standalone companion to embedding_ablation.py for the case where the ablation
was run without --pca-components (or with a very large dataset where re-running
the full extraction is impractical).

Fits PCA on a random subsample of patches (default 1M rows), then transforms
the full dataset in batches to stay within memory limits.  All outputs match
the format that unified_probing.py expects.

Outputs written to <ablation-dir>/:
    patch_residuals_pca.npy            (N, n_keep, n_components)  float32
    pca_components.npy                 (n_components, D)           float32
    pca_explained_variance_ratio.npy   (n_components,)             float32
    pca_scree.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ablation-dir", type=Path, required=True,
                   help="Output directory from embedding_ablation.py "
                        "(must contain patch_residuals.npy).")
    p.add_argument("--n-components", type=int, default=50,
                   help="Number of PCA components (default 50).")
    p.add_argument("--n-fit-samples", type=int, default=1_000_000,
                   help="Rows subsampled from the full (N×n_keep, D) matrix "
                        "for PCA fitting (default 1M).  Fitting on the full "
                        "42M-row matrix is ~42× slower with negligible "
                        "accuracy gain; 1M rows is statistically sufficient "
                        "to capture the covariance structure.")
    p.add_argument("--transform-batch-size", type=int, default=50,
                   help="Number of samples to transform at a time "
                        "(default 50).  Each batch uses ~650 MB at "
                        "D=1280, n_keep=65536.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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
                xytext=(k + 1, thresh - 0.05),
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    residuals_path = args.ablation_dir / "patch_residuals.npy"
    if not residuals_path.exists():
        raise FileNotFoundError(f"patch_residuals.npy not found in {args.ablation_dir}")

    # --- Load residuals -------------------------------------------------------
    # At 204 GB (650, 65536, 1280) float32 this fits on a 487 GB node.
    # Use mmap_mode so the OS pages data on demand during the fit subsample,
    # then loads sequentially during the transform batches.
    print(f"[1/4] Loading residuals from {residuals_path} ...")
    residuals = np.load(residuals_path, mmap_mode="r")
    N, n_keep, D = residuals.shape
    n_total = N * n_keep
    print(f"      shape: ({N}, {n_keep}, {D})  total patches: {n_total:,}")

    # --- Subsample rows for fitting -------------------------------------------
    n_fit = min(args.n_fit_samples, n_total)
    print(f"\n[2/4] Subsampling {n_fit:,} / {n_total:,} rows for PCA fit ...")
    idx = rng.choice(n_total, size=n_fit, replace=False)
    idx.sort()  # sorted access is faster on mmap

    # Flatten residuals to (n_total, D) as a view, then index the subsample.
    flat_all = residuals.reshape(n_total, D)   # view; no copy
    fit_data = flat_all[idx].copy()            # (n_fit, D) — copy to RAM for fit

    # --- Fit PCA --------------------------------------------------------------
    print(f"\n[3/4] Fitting PCA({args.n_components}) on {n_fit:,} rows ...")
    pca = PCA(n_components=args.n_components, svd_solver="randomized",
              random_state=args.seed)
    pca.fit(fit_data)
    del fit_data  # free ~500 MB

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"      Top {args.n_components} PCs cumulative variance: "
          f"{cumvar[-1]*100:.2f}%")
    for thresh in (0.5, 0.9, 0.95, 0.99):
        if cumvar[-1] >= thresh:
            k = int(np.searchsorted(cumvar, thresh)) + 1
            print(f"      {int(thresh*100)}% variance: {k} components")

    # Save PCA basis and variance immediately so they're available even if
    # the transform step is interrupted.
    np.save(args.ablation_dir / "pca_components.npy",
            pca.components_.astype(np.float32))
    np.save(args.ablation_dir / "pca_explained_variance_ratio.npy",
            pca.explained_variance_ratio_.astype(np.float32))
    plot_pca_scree(pca.explained_variance_ratio_,
                   args.ablation_dir / "pca_scree.png")
    print("      Saved pca_components.npy, pca_explained_variance_ratio.npy, "
          "pca_scree.png")

    # --- Transform full dataset in batches ------------------------------------
    # Output shape (N, n_keep, n_components) stored fully in RAM (~8.5 GB).
    print(f"\n[4/4] Transforming all {N} samples in batches of "
          f"{args.transform_batch_size} ...")
    out = np.empty((N, n_keep, args.n_components), dtype=np.float32)

    bs = args.transform_batch_size
    for start in tqdm(range(0, N, bs), desc="Transforming batches"):
        end = min(start + bs, N)
        batch = residuals[start:end].reshape((end - start) * n_keep, D)
        transformed = pca.transform(batch).astype(np.float32)
        out[start:end] = transformed.reshape(end - start, n_keep, args.n_components)

    out_path = args.ablation_dir / "patch_residuals_pca.npy"
    np.save(out_path, out)
    print(f"\nSaved {out_path}  shape={out.shape}  dtype={out.dtype}")


if __name__ == "__main__":
    main()
