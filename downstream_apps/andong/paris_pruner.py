"""PARIS — Pruning Algorithm via the Representer theorem for Imbalanced Scenarios.

Implementation of Algorithm 1 from the PARIS spec. The algorithm:
  1. Assumes a trained NN split into φ (feature extractor) + final Linear (w, b).
  2. Treats predictions on validation as kernel-ridge-style sums over training.
  3. Iteratively removes training samples that most reduce validation loss on
     the hardest (largest-residual) validation point.
  4. Uses Cholesky rank-one downdate to update the inverse-Gram matrix in
     O(D²) per iteration instead of O(D³) recomputation.

Usage sketch (multi-output / sequence models like our V12/V13 require
extracting per-sample penultimate features and reducing to a single regression
target — see _MockModel + main() at the bottom for the canonical interface).

Caveat: the deletion-residual formula
    ΔL_{\k}(v) = 2·r_v·S_{v,k} + S_{v,k}²
is an approximation that assumes removing sample k drops its contribution
S_{v,k} = (φ_v·φ_k)(φ_k·w*) to the prediction, with w* held fixed. The exact
LOO change in KRR additionally includes a regularization-mediated update to
w*. For ranking the most-harmful k, this approximation is the same one used
in the influence-function literature and works well in practice. Reported
ΔL values are heuristic, not exact validation-loss differences.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================
# Cholesky rank-one downdate
# ============================================================
def cholesky_downdate(L: np.ndarray, x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return L_new lower-triangular such that  L_new L_new^T = L L^T − x x^T.

    Uses sequential Givens-style rotations. O(D²) work.

    Args:
        L: (D, D) lower-triangular Cholesky factor of A (positive definite).
        x: (D,) downdate vector.
        eps: numerical floor for positive-definiteness check.

    Returns:
        L_new: (D, D) lower-triangular Cholesky factor of  A − x x^T.

    Raises:
        ValueError: if A − x x^T is not positive definite (||L⁻¹ x||² ≥ 1).
    """
    L = L.copy().astype(np.float64)
    x = x.astype(np.float64).copy()
    D = L.shape[0]

    # Solve L · p = x by forward substitution.
    p = np.empty(D, dtype=np.float64)
    for i in range(D):
        s = x[i] - L[i, :i] @ p[:i]
        p[i] = s / L[i, i]

    # Positive-definiteness check: A − xxᵀ ≻ 0  ⇔  ||p||² < 1.
    pp = float(p @ p)
    if pp >= 1.0 - eps:
        raise ValueError(f"Downdate would violate positive-definiteness "
                         f"(||L⁻¹x||² = {pp:.6g} ≥ 1).")
    rho = math.sqrt(1.0 - pp)

    # Apply hyperbolic rotations bottom-up to fold p into rho while updating L.
    # We construct R such that  [L 0; pᵀ rho]ᵀ R = [L_new 0; 0 ?]ᵀ.
    # Equivalently: for each row i from D-1 down to 0, rotate (rho, p_i) → (rho', 0)
    # and apply the same rotation to (L[i, :i+1], 0_row).
    aug = np.zeros(D, dtype=np.float64)
    for i in range(D - 1, -1, -1):
        a = rho
        b = p[i]
        h = math.hypot(a, b)  # = sqrt(a² + b²)
        # Hyperbolic-style rotation that preserves "L Lᵀ − xxᵀ" structure:
        # c = a / h,  s = b / h   (regular Givens; valid here because of the
        # earlier check pp < 1 which guarantees we never need a true hyperbolic
        # rotation — the augmented system is well-posed.)
        if h < eps:
            continue
        c = a / h
        s = b / h
        rho = c * rho + s * 0.0   # rho stays as h (positive)
        rho = h
        # Apply rotation across row i of L and the auxiliary row `aug` up to col i
        L_row_i = L[i, :i + 1].copy()
        aug_i  = aug[:i + 1].copy()
        L[i, :i + 1] = c * L_row_i + s * aug_i
        aug[:i + 1]  = c * aug_i - s * L_row_i
        # Update the (i)-th component of p to 0 (folded into rho); no explicit op needed.
    return L


def cholesky_downdate_naive(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fallback / sanity-check downdate: form A_new = L Lᵀ − xxᵀ and re-Cholesky.
    O(D³) per call. Use this for small D (≤ 256) or to validate the Givens version.
    """
    A_new = L @ L.T - np.outer(x, x)
    # Symmetrize to combat numerical drift.
    A_new = 0.5 * (A_new + A_new.T)
    return np.linalg.cholesky(A_new)


# ============================================================
# Lambda estimator
# ============================================================
def estimate_lambda(Phi: np.ndarray, Y_c: np.ndarray, w_NN: np.ndarray,
                    lambda_min: float = 1e-5) -> float:
    """Closed-form estimate of the ridge regularization λ implied by a trained
    linear head. From the normal equations:
        (ΦᵀΦ + λI) w_NN = ΦᵀY_c
        λ w_NN = ΦᵀY_c − ΦᵀΦ w_NN
    Take inner product with w_NN:
        λ ||w_NN||² = w_NNᵀ (ΦᵀY_c − ΦᵀΦ w_NN)
    """
    A = Phi.T @ Phi
    b = Phi.T @ Y_c
    num = float(w_NN @ (b - A @ w_NN))
    den = float(w_NN @ w_NN)
    if den < 1e-12:
        return lambda_min
    lam = num / den
    if not np.isfinite(lam) or lam <= 0:
        return lambda_min
    return float(max(lam, lambda_min))


# ============================================================
# PARIS pruner
# ============================================================
class PARISPruner:
    """Iteratively prune K samples from a training set using the representer-
    theorem influence formula plus efficient Cholesky downdates.

    Interface (you provide):
        phi: callable mapping a batch of inputs to penultimate features (B, D)
        w_NN: (D,) final-layer weights (single-output regression)
        b_NN: float bias of the final layer
        train_loader: yields (x, y) pairs of the training set, in a stable
            order so that prune-indices map back to dataset rows
        val_loader: same, for the validation set
        device: torch device for feature extraction (CPU fine for D ≤ 1024)
        prune_fraction: fraction p of training set to prune (K = ⌊p·N⌋)

    Returns:
        prune_idx: list of training indices selected for removal, in deletion order
        keep_idx: list of remaining training indices
    """

    def __init__(self,
                 phi: Callable[[torch.Tensor], torch.Tensor],
                 w_NN: torch.Tensor,
                 b_NN: float,
                 prune_fraction: float = 0.10,
                 device: str = 'cpu',
                 downdate_impl: str = 'givens',
                 verbose: bool = True,
                 lambda_override: Optional[float] = None):
        assert downdate_impl in ('givens', 'naive')
        self.phi = phi
        self.w_NN = w_NN.detach().cpu().numpy().astype(np.float64).flatten()
        self.b_NN = float(b_NN)
        self.prune_fraction = float(prune_fraction)
        self.device = device
        self.downdate_impl = downdate_impl
        self.verbose = verbose
        self.lambda_override = lambda_override

    # ----- feature extraction -----
    @torch.no_grad()
    def _extract(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Run φ over the entire loader. Returns (Φ, Y) as numpy arrays.
        Loader yields (batch_input, y).  batch_input may be a Tensor OR a tuple of
        Tensors — device movement is delegated to self.phi.  y is a scalar target
        per sample (single-output regression).
        """
        Phi_chunks = []
        Y_chunks = []
        for batch in loader:
            x, y = batch[0], batch[1]
            if isinstance(x, torch.Tensor):
                x = x.to(self.device)
            f = self.phi(x).detach().cpu().numpy().astype(np.float64)
            Phi_chunks.append(f.reshape(f.shape[0], -1))
            Y_chunks.append(y.detach().cpu().numpy().astype(np.float64).flatten())
        return np.concatenate(Phi_chunks, axis=0), np.concatenate(Y_chunks, axis=0)

    # ----- inner loop -----
    def prune(self,
              train_loader: DataLoader,
              val_loader: DataLoader) -> dict:
        """Run the PARIS algorithm. Returns dict with prune_idx, keep_idx, debug stats."""
        log = print if self.verbose else (lambda *a, **kw: None)

        # ---- 1. Extract features ----
        log("[PARIS] Extracting features for training set...")
        Phi_tr, Y_tr = self._extract(train_loader)
        Y_tr_c = Y_tr - self.b_NN

        log("[PARIS] Extracting features for validation set...")
        Phi_val, Y_val = self._extract(val_loader)
        Y_val_c = Y_val - self.b_NN
        N, D = Phi_tr.shape
        N_val = Phi_val.shape[0]
        log(f"[PARIS] N_train={N}  N_val={N_val}  feature_dim={D}")

        # ---- 2. Estimate (or override) λ ----
        if self.lambda_override is not None:
            lam = float(self.lambda_override)
            log(f"[PARIS] Using fixed λ = {lam:.4e} (override; skipping closed-form estimator)")
        else:
            lam = estimate_lambda(Phi_tr, Y_tr_c, self.w_NN)
            log(f"[PARIS] Estimated λ = {lam:.4e}")

        # ---- 3. Build A = ΦᵀΦ + λI and Cholesky ----
        A = Phi_tr.T @ Phi_tr + lam * np.eye(D)
        L = np.linalg.cholesky(A)              # A = L Lᵀ
        # ---- 4. Solve for w* by two triangular solves on L Lᵀ w* = Φᵀ Y_c ----
        Phi_T_Y = Phi_tr.T @ Y_tr_c
        w_star = self._chol_solve(L, Phi_T_Y)
        log(f"[PARIS] w* solved (||w*||={np.linalg.norm(w_star):.4f})")

        # α = canonical KRR dual coefficients: α = (Y_c − Φ w*) / λ
        #   so that  w* = Φᵀ α   and   Φ_val w* = Φ_val Φᵀ α = T α.
        # (The spec's "α = Φ w*" gives the predicted training residual, which
        # does NOT satisfy Y_hat = T α — see note at top of file. Using the
        # canonical form makes the deletion-residual formula a true LOO
        # approximation.)
        alpha = (Y_tr_c - Phi_tr @ w_star) / lam  # (N,)

        # T_{i,j} = φ_{val,i} · φ_{train,j}; shape (N_val, N)
        T = Phi_val @ Phi_tr.T

        # ---- 5. Inner loop: prune K samples ----
        K = int(math.floor(self.prune_fraction * N))
        log(f"[PARIS] Pruning K = {K} samples (p={self.prune_fraction})")

        # alive: boolean mask over original training indices; T and α columns
        # are indexed into the *current* active set, so we also track an
        # original-index map.
        active = list(range(N))     # original index for each current column of T
        prune_order = []             # list of original indices in deletion order

        for it in range(K):
            # Scaled influence S = T ⊙ αᵀ   (each column j scaled by α_j)
            S = T * alpha[np.newaxis, :]      # (N_val, len(active))
            # Validation predictions and residuals
            Y_hat_val = S.sum(axis=1)
            r_val = Y_val_c - Y_hat_val       # using centered targets
            # Hardest validation point
            v_star = int(np.argmax(r_val ** 2))
            r_v = r_val[v_star]
            # Divergence guard: residuals should stay on the scale of the
            # initial fit; explosive growth means the iterative state (L, w*,
            # α) has drifted and the ranking is no longer influence.
            if it == 0:
                r_max0 = float(np.abs(r_val).max()) + 1e-12
            elif float(np.abs(r_val).max()) > 1e3 * r_max0:
                raise RuntimeError(
                    f"PARIS diverged at iter {it}: max|r_val| grew "
                    f"{float(np.abs(r_val).max())/r_max0:.2e}x over the initial fit")
            # Deletion losses for each remaining training sample
            S_row = S[v_star, :]              # (len(active),)
            delta_L = 2.0 * r_v * S_row + S_row ** 2
            k_local = int(np.argmin(delta_L))
            k_global = active[k_local]
            prune_order.append(k_global)

            if it < 3 or (it + 1) % max(1, K // 10) == 0 or it == K - 1:
                log(f"  [PARIS] iter {it+1:>4}/{K}: hardest v={v_star} (r²={r_v**2:.3f}), "
                    f"prune k_global={k_global} (ΔL={delta_L[k_local]:.4f})")

            # ---- 6. Efficient update ----
            phi_k = Phi_tr[k_local]
            y_k_c = Y_tr_c[k_local]
            try:
                if self.downdate_impl == 'givens':
                    L = cholesky_downdate(L, phi_k)
                else:
                    L = cholesky_downdate_naive(L, phi_k)
            except ValueError as e:
                # If downdate fails, refactor from scratch
                log(f"  [PARIS] downdate failed ({e}); refactoring")
                Phi_T_Y -= phi_k * y_k_c              # update RHS too (local index)
                A_new = A - np.outer(phi_k, phi_k)
                A = A_new
                L = np.linalg.cholesky(0.5 * (A_new + A_new.T))
            else:
                # Update RHS Φᵀ Y_c by removing the contribution of sample k
                Phi_T_Y = Phi_T_Y - phi_k * y_k_c
                A = A - np.outer(phi_k, phi_k)        # keep A in sync for fallback

            # Mark removed: drop column k_local from T, row k_local from alpha
            # and the corresponding row from Phi_tr / Y_tr_c.
            mask = np.ones(len(active), dtype=bool); mask[k_local] = False
            T = T[:, mask]
            Phi_tr = Phi_tr[mask, :]
            Y_tr_c = Y_tr_c[mask]
            active.pop(k_local)

            # Recompute w* and α with the downdated L.  α must stay in the
            # canonical dual form used at initialization — recomputing it as
            # Φ w* (predicted values) is a different quantity at a different
            # scale and makes the loop state diverge over thousands of
            # deletions (observed: val r² growing to ~1e13 on fold-20 data).
            w_star = self._chol_solve(L, Phi_T_Y)
            alpha = (Y_tr_c - Phi_tr @ w_star) / lam

        keep_idx = sorted(active)
        return {
            'prune_idx': prune_order,
            'keep_idx': keep_idx,
            'lambda': lam,
            'K': K,
            'N_train_original': N,
            'N_val': N_val,
            'feature_dim': D,
        }

    # ----- helper -----
    @staticmethod
    def _chol_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve L Lᵀ x = b via two triangular solves."""
        y = np.linalg.solve(np.tril(L), b)
        x = np.linalg.solve(np.tril(L).T, y)
        return x


# ============================================================
# Mock execution script
# ============================================================
class _MockNet(nn.Module):
    """Tiny model with explicit phi + linear head, for the demo."""
    def __init__(self, in_dim=20, feat_dim=16):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(),
                                  nn.Linear(32, feat_dim), nn.ReLU())
        self.head = nn.Linear(feat_dim, 1)  # single-output regression

    def forward(self, x):
        return self.head(self.phi(x)).squeeze(-1)


def _mock_demo():
    """Synthetic imbalanced regression demo: 95% near-zero targets, 5% extreme."""
    torch.manual_seed(0); np.random.seed(0)
    N_train, N_val, D_in = 2000, 400, 20
    X_tr = torch.randn(N_train, D_in)
    Y_tr = torch.randn(N_train) * 0.3
    # Inject 5% "storm" samples with large targets
    storm_idx = np.random.choice(N_train, size=int(0.05*N_train), replace=False)
    Y_tr[storm_idx] += torch.randn(len(storm_idx)) * 3 + 4
    X_val = torch.randn(N_val, D_in)
    Y_val = torch.randn(N_val) * 0.3
    val_storm = np.random.choice(N_val, size=int(0.05*N_val), replace=False)
    Y_val[val_storm] += torch.randn(len(val_storm)) * 3 + 4

    train_loader = DataLoader(list(zip(X_tr, Y_tr)), batch_size=64, shuffle=False)
    val_loader   = DataLoader(list(zip(X_val, Y_val)), batch_size=64, shuffle=False)

    # Quick train (5 epochs)
    model = _MockNet(in_dim=D_in, feat_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for ep in range(5):
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            ((pred - yb) ** 2).mean().backward()
            opt.step()

    # PARIS
    pruner = PARISPruner(
        phi=model.phi,
        w_NN=model.head.weight,
        b_NN=float(model.head.bias),
        prune_fraction=0.10,
        downdate_impl='naive',     # use naive (O(D³)) for tiny D — known correct
        verbose=True,
    )
    out = pruner.prune(train_loader, val_loader)
    print()
    print(f"Done. λ = {out['lambda']:.4e}, K = {out['K']}")
    print(f"First 10 pruned (original indices): {out['prune_idx'][:10]}")
    print(f"How many of the {len(storm_idx)} injected storm samples were pruned? "
          f"{sum(1 for i in out['prune_idx'] if i in set(storm_idx))} "
          f"(if PARIS is working, this should be LOW — we want to keep storm samples)")


if __name__ == "__main__":
    _mock_demo()
