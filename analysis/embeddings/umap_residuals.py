#!/usr/bin/env python3
"""
Run UMAP/HDBSCAN on Surya patch residuals (post per-position mean ablation).

Loads outputs from embedding_ablation.py, projects residuals to 2D with UMAP,
clusters with HDBSCAN, and produces the same diagnostic figures that
embedding_analysis.py produces — using its plotting helpers — but on the
ablated representation.

Compared to the original embedding_analysis.py UMAP step:
  * Input is residuals X[t,i] − μ[i] instead of raw X[t,i].
  * UMAP knobs default to local-neighborhood-preserving regime
    (min_dist=0.1, n_neighbors=30) instead of the original global-structure
    knobs (min_dist=0.99, n_neighbors=200).  The original knobs were
    exaggerating the position-encoding 2-torus; for content-driven structure
    we want UMAP looking at local neighborhoods.
  * HDBSCAN min_cluster_size defaults to 500 (vs the original 2) so we get a
    handful of meaningful content clusters instead of thousands of singleton
    fragments.

Outputs in --out-dir:
  patch_residuals_2d.npy        (N*n_keep, 2)  UMAP coordinates.
  patch_residuals_clusters.npy  (N*n_keep,)    HDBSCAN labels.
  umap_by_mask_class.png        UMAP colored by Coronal Hole / Quiet Sun /
                                 Active Region / NA.
  umap_by_cluster.png           UMAP colored by HDBSCAN cluster.
  umap_by_position.png          UMAP colored by patch (row, col).  KEY
                                 DIAGNOSTIC: should look like noise; if there
                                 is still a smooth color gradient, ablation
                                 didn't fully decouple position.
  spatial_mask_labels.png       Spatial mask labels for reference.
  spatial_cluster_labels.png    Spatial cluster labels — the test of whether
                                 residual clusters track physical features.
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

# Reuse the plotting helpers and clusterer from the original analysis script.
from embedding_analysis import (
    PATCH_GRID,
    MASK_CLASS_NAMES,
    cluster_embeddings,
    plot_umap_by_mask_class,
    plot_umap_by_cluster,
    plot_umap_by_position,
    plot_spatial_mask_labels,
    plot_spatial_cluster_labels,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ablation-dir", type=Path, required=True,
                   help="Directory produced by embedding_ablation.py.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: <ablation-dir>/umap_residuals/).")
    p.add_argument("--use-pca", action="store_true",
                   help="Use patch_residuals_pca.npy as input instead of full residuals.")
    p.add_argument("--n-neighbors", type=int, default=30,
                   help="UMAP n_neighbors (default 30; the original script used 200).")
    p.add_argument("--min-dist", type=float, default=0.1,
                   help="UMAP min_dist (default 0.1; the original script used 0.99).")
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=500,
                   help="HDBSCAN min_cluster_size (default 500).")
    p.add_argument("--hdbscan-min-samples", type=int, default=50,
                   help="HDBSCAN min_samples (default 50).")
    p.add_argument("--reassign-noise", action="store_true",
                   help="HDBSCAN only: after clustering, reassign every noise "
                        "point (-1) to its nearest cluster centroid in the "
                        "clustering feature space. Yields a hard label for "
                        "every point (no -1 class). The unmodified labels are "
                        "still saved alongside.")
    p.add_argument("--cluster-on", choices=["raw", "pca", "umap2d"], default="umap2d",
                   help="Feature space for clustering. 'raw' = 1280-d residuals "
                        "(suffers high-dim distance concentration; often fails). "
                        "'pca' = 50-PC residuals. 'umap2d' = the 2D UMAP "
                        "projection (default; most reliable in practice).")
    p.add_argument("--cluster-method", choices=["hdbscan", "kmeans", "gmm"], default="kmeans",
                   help="Clustering algorithm. 'kmeans' (default) gives a fixed "
                        "number of clusters with no noise label and is the right "
                        "tool when you want to ask 'do these N clusters track "
                        "physical labels?'. 'hdbscan' finds variable-count "
                        "clusters and labels low-density points as noise. "
                        "'gmm' fits Gaussian mixtures (handles elongated modes "
                        "K-means can't) and can sweep BIC to pick K automatically.")
    p.add_argument("--n-clusters", type=int, default=10,
                   help="K for K-means or GMM (ignored if cluster_method=hdbscan, "
                        "or if --gmm-bic-sweep is used).")
    p.add_argument("--gmm-bic-sweep", type=str, default=None,
                   help="GMM only: comma-separated K values to sweep "
                        "(e.g., '3,5,7,10,15'). Saves BIC plot, picks lowest.")
    p.add_argument("--gmm-covariance", choices=["full", "tied", "diag", "spherical"],
                   default="full",
                   help="GMM covariance type. 'full' allows arbitrary-shape "
                        "ellipses (most flexible). Default: full.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = args.ablation_dir / "umap_residuals"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.out_dir}")

    # --- 1. Load arrays -----------------------------------------------------
    print("\n[1/3] Loading residuals and labels ...")
    if args.use_pca:
        residuals_path = args.ablation_dir / "patch_residuals_pca.npy"
        residuals = np.load(residuals_path)
        print(f"  PCA residuals: {residuals.shape} (from {residuals_path.name})")
    else:
        residuals_path = args.ablation_dir / "patch_residuals.npy"
        residuals = np.load(residuals_path)
        print(f"  Raw residuals: {residuals.shape} (from {residuals_path.name})")

    positions  = np.load(args.ablation_dir / "patch_residuals_positions.npy")  # (n_keep, 2)
    labels     = np.load(args.ablation_dir / "patch_mask_labels.npy")           # (N, n_keep)

    N, n_keep, D = residuals.shape

    # Reconstruct the keep_idx that the original plotting helpers expect
    # (flat raster index = row * PATCH_GRID + col).
    rows = positions[:, 0].astype(np.int64)
    cols = positions[:, 1].astype(np.int64)
    keep_idx = (rows * PATCH_GRID + cols).astype(np.int64)

    # Flatten for UMAP/HDBSCAN.  Residuals are the input feature space;
    # positions are broadcast across samples to produce a per-point coord.
    flat_embs   = residuals.reshape(-1, D)
    flat_labels = labels.reshape(-1)
    flat_rows   = np.broadcast_to(rows[None, :], (N, n_keep)).reshape(-1)
    flat_cols   = np.broadcast_to(cols[None, :], (N, n_keep)).reshape(-1)

    print(f"  Total points for UMAP : {len(flat_embs):,}")
    for cls_id, cls_name in sorted(MASK_CLASS_NAMES.items()):
        n = int((flat_labels == cls_id).sum())
        print(f"    {cls_name:14s}: {n:>10,} patches")

    # --- 2. UMAP ------------------------------------------------------------
    print(f"\n[2/3] Running UMAP "
          f"(n_neighbors={args.n_neighbors}, min_dist={args.min_dist}) ...")
    try:
        from cuml import manifold as umap_lib
    except ImportError:
        raise ImportError("cuML required: conda install -c rapidsai cuml")

    reducer = umap_lib.UMAP(
        n_neighbors  = args.n_neighbors,
        min_dist     = args.min_dist,
        n_components = 2,
        random_state = args.seed,
        verbose      = True,
    )
    emb2d = reducer.fit_transform(flat_embs)
    np.save(args.out_dir / "patch_residuals_2d.npy", emb2d)

    # --- 3. Clustering + plots ---------------------------------------------
    # Pick the feature space:
    if args.cluster_on == "raw":
        cluster_input = flat_embs
        cluster_space = f"raw residuals ({D}-d)"
    elif args.cluster_on == "pca":
        pca_path = args.ablation_dir / "patch_residuals_pca.npy"
        cluster_input = np.load(pca_path).reshape(-1, np.load(pca_path).shape[-1])
        cluster_space = f"PCA residuals ({cluster_input.shape[1]}-d)"
    else:  # umap2d
        cluster_input = emb2d
        cluster_space = "2D UMAP projection"

    print(f"\n[3/3] Clustering ({args.cluster_method}) on {cluster_space} ...")

    if args.cluster_method == "hdbscan":
        print(f"  HDBSCAN: min_cluster_size={args.hdbscan_min_cluster_size}, "
              f"min_samples={args.hdbscan_min_samples}")
        cluster_labels = cluster_embeddings(
            cluster_input,
            min_cluster_size = args.hdbscan_min_cluster_size,
            min_samples      = args.hdbscan_min_samples,
            cluster_selection_method = "leaf",
            cluster_selection_epsilon = 1.0,
            alpha = 1.0
        )

        if args.reassign_noise:
            noise_mask = cluster_labels == -1
            n_noise    = int(noise_mask.sum())
            uniq       = np.unique(cluster_labels[~noise_mask])

            if len(uniq) == 0:
                print("  reassign-noise: nothing to do — every point is noise. "
                      "Try looser HDBSCAN parameters first.")
            elif n_noise == 0:
                print("  reassign-noise: nothing to do — no noise points.")
            else:
                # Save the unmodified labels before overwriting.
                np.save(args.out_dir / "patch_residuals_clusters_with_noise.npy",
                        cluster_labels.copy())

                print(f"  reassign-noise: {n_noise:,} of {len(cluster_labels):,} "
                      f"points ({100*n_noise/len(cluster_labels):.1f}%) → "
                      f"nearest of {len(uniq)} centroids")

                # Centroid per cluster in the clustering feature space.
                centroids = np.stack([
                    cluster_input[cluster_labels == k].mean(axis=0).astype(np.float32)
                    for k in uniq
                ])

                # Chunked nearest-centroid assignment to bound peak memory.
                # noise_pts: (M, F);  centroids: (K, F)  → distances: (M, K).
                # We process in chunks of CHUNK rows to keep the temporary
                # array small even when M and F are both large.
                noise_pts = cluster_input[noise_mask].astype(np.float32)
                CHUNK = 50_000
                assigned = np.empty(n_noise, dtype=np.int32)
                # Pre-compute centroid squared norms for the (a-b)² expansion.
                cent_sq  = (centroids ** 2).sum(axis=1)
                for start in range(0, n_noise, CHUNK):
                    chunk = noise_pts[start:start + CHUNK]
                    # ‖x − c‖² = ‖x‖² + ‖c‖² − 2·x·cᵀ
                    # ‖x‖² is constant per row, so we omit it for the argmin.
                    d2 = cent_sq[None, :] - 2.0 * chunk @ centroids.T
                    assigned[start:start + CHUNK] = uniq[d2.argmin(axis=1)]

                cluster_labels = cluster_labels.copy()
                cluster_labels[noise_mask] = assigned
    elif args.cluster_method == "kmeans":
        from cuml.cluster import KMeans
        print(f"  K-means: n_clusters={args.n_clusters}")
        km = KMeans(
            n_clusters   = args.n_clusters,
            random_state = args.seed,
            max_iter     = 300,
        )
        cluster_labels = km.fit_predict(cluster_input).astype(np.int32)
        n_per = np.bincount(cluster_labels)
        print(f"  K-means: cluster sizes = {sorted(n_per.tolist(), reverse=True)}")

    else:  # gmm
        from sklearn.mixture import GaussianMixture
        from sklearn.utils import resample

        # GMM is sklearn-CPU. At 1M points, full-covariance EM is slow but
        # tractable (~minutes); fit on a subsample and predict on the full set.
        n_fit = min(200_000, len(cluster_input))
        rng_fit = np.random.default_rng(args.seed)
        fit_idx = rng_fit.choice(len(cluster_input), size=n_fit, replace=False)
        fit_data = cluster_input[fit_idx]

        if args.gmm_bic_sweep:
            ks = [int(k.strip()) for k in args.gmm_bic_sweep.split(",")]
            print(f"  GMM BIC sweep over K = {ks}  (fit subsample n={n_fit:,})")
            bics = []
            for k in ks:
                gmm = GaussianMixture(
                    n_components    = k,
                    covariance_type = args.gmm_covariance,
                    random_state    = args.seed,
                    max_iter        = 200,
                    n_init          = 1,
                )
                gmm.fit(fit_data)
                bic = gmm.bic(fit_data)
                bics.append(bic)
                print(f"    K={k:3d}: BIC = {bic:.2f}")

            # Save sweep plot.
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(ks, bics, "o-")
            ax.set_xlabel("K (number of components)")
            ax.set_ylabel("BIC (lower = better)")
            ax.set_title("GMM BIC sweep")
            best_k = ks[int(np.argmin(bics))]
            ax.axvline(best_k, color="red", linestyle="--",
                       label=f"best K = {best_k}")
            ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(args.out_dir / "gmm_bic_sweep.png", dpi=150)
            plt.close(fig)
            print(f"  saved -> {args.out_dir/'gmm_bic_sweep.png'}")
            print(f"  Selected K={best_k} (lowest BIC)")
            n_components = best_k
        else:
            n_components = args.n_clusters
            print(f"  GMM: n_components={n_components}, "
                  f"covariance={args.gmm_covariance}, fit subsample n={n_fit:,}")

        gmm = GaussianMixture(
            n_components    = n_components,
            covariance_type = args.gmm_covariance,
            random_state    = args.seed,
            max_iter        = 200,
            n_init          = 3,
        )
        gmm.fit(fit_data)
        cluster_labels = gmm.predict(cluster_input).astype(np.int32)
        n_per = np.bincount(cluster_labels)
        print(f"  GMM: cluster sizes = {sorted(n_per.tolist(), reverse=True)}")

    np.save(args.out_dir / "patch_residuals_clusters.npy", cluster_labels)

    title_suffix = (
        f"\nResiduals (post position-mean ablation) — "
        f"{N} samples × {n_keep:,} patches = {len(flat_embs):,} points"
    )

    print("\nGenerating figures ...")
    plot_umap_by_mask_class(
        emb2d, flat_labels,
        args.out_dir / "umap_by_mask_class.png", title_suffix,
    )
    plot_umap_by_cluster(
        emb2d, cluster_labels,
        args.out_dir / "umap_by_cluster.png", title_suffix,
    )
    plot_umap_by_position(
        emb2d, flat_rows, flat_cols, PATCH_GRID,
        args.out_dir / "umap_by_position.png", title_suffix,
    )
    plot_spatial_mask_labels(
        labels, keep_idx, PATCH_GRID,
        args.out_dir / "spatial_mask_labels.png",
        f"\n({N} samples)",
    )
    plot_spatial_cluster_labels(
        cluster_labels, N, keep_idx, PATCH_GRID,
        args.out_dir / "spatial_cluster_labels.png",
        f"\n({N} samples)",
    )

    print(f"\nDone. Outputs in {args.out_dir}/")
    print("\nKey diagnostics:")
    print("  umap_by_position.png  — should look ~random in color now")
    print("                          (smooth gradient = position not fully removed)")
    print("  umap_by_mask_class.png — ARs/CHs/QS should be distinguishable")
    print("                           (and not appear in 4 mirror copies as before)")


if __name__ == "__main__":
    main()