"""PARIS pruning test on LIVE-Ap (V14-AGC), fold 20 (Gannon, manuscript Event #27).

Protocol (preflight: experiments_log.md 2026-07-15 02:46 UTC):
  1. Rebuild the exact canonical fold-20 split (seed-42 shuffle, 85/15).
  2. Load the canonical fold-20 checkpoint; extract phi = decoder GRU hidden
     at lead index 23 (72 h) for train and val splits; w,b = dec_proj.
  3. PARIS-prune 10% of the 85% train split using the 15% val split as the
     PARIS validation set (NOT the held-out event — leak-free).
     lambda override 1e-2 (matches the 2026-06 V12 diagnostics).
  4. Retrain with the otherwise-identical canonical protocol (30 ep, seed 42,
     PDF sampler weights subset from the pool lookup) on the kept samples.
  5. Score with v14_agc_loocv_ensemble.run_fold and print vs canonical + SWPC.

Output: runs/v14_agc_ap_emu_paris/fold_20_Event_22/   (canonical dir untouched)
        runs/v14_agc_ap_emu_paris/fold_20_Event_22/paris_prune_indices.csv
"""
from __future__ import annotations
import os, sys, json, argparse, time

os.environ.setdefault('EMBEDDING_PATH_OVERRIDE',
    '/media/faraday/andong/Dataspace/GONG_NN/Data/Model_B_AIA_GONG_C3_2011_2025.h5')
os.environ.setdefault('IMG_SLICE_MODE', 'all_three')
os.environ.setdefault('EPOCHS_OVERRIDE', '30')
os.environ.setdefault('CACHE_EMBEDDINGS', '1')     # I/O path only; same data

ap = argparse.ArgumentParser()
ap.add_argument('--gpu', type=int, default=7)
ap.add_argument('--fold', type=int, default=20)
ap.add_argument('--prune-fraction', type=float, default=0.10)
ap.add_argument('--lam', type=float, default=1e-2)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--prune-only', action='store_true',
                help='stop after PARIS + diagnostics (no retrain)')
args = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong')
import paris_agc_loocv_fold_v14_ap_emu as W
from paris_pruner import PARISPruner, estimate_lambda
from dataset_leakfree import GatedDatasetLeakFree
from dataset_leakfree_ap_emu import ApEmulatorDataset
from dataset_leakfree_ap import AP_STORM_THRESHOLD

ROOT = '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong'
PARIS_ROOT = f'{ROOT}/runs/v14_agc_ap_emu_paris'
LEAD = W.LEAD_72H          # 23
DEVICE = 'cuda:0'

log = lambda *a: print('[paris-f20]', *a, flush=True)


@torch.no_grad()
def extract_phi(model, ds, indices, lead=LEAD, bs=256):
    """phi = decoder GRU hidden at `lead`; also returns target and head pred."""
    model.eval()
    loader = DataLoader(Subset(ds, indices), batch_size=bs, shuffle=False,
                        num_workers=4, pin_memory=True)
    Phis, Ys, Preds = [], [], []
    for batch in loader:
        x_img, x_anemo, x_dst, x_ap_hist, y_ap = [b.to(DEVICE) for b in batch[:5]]
        c = model.encode_context(x_img, x_anemo, x_dst, x_ap_hist)
        h = c
        ap_prev = x_ap_hist[:, -1]
        for k in range(model.forecast_horizon):
            inp = torch.cat([c, ap_prev.unsqueeze(-1)], dim=-1)
            h = model.dec_cell(inp, h)
            ap_pred = model.dec_proj(h).squeeze(-1)
            if k == lead:
                Phis.append(h.cpu()); Preds.append(ap_pred.cpu())
                break
            ap_prev = ap_pred
        Ys.append(y_ap[:, lead].cpu())
    return torch.cat(Phis), torch.cat(Ys), torch.cat(Preds)


def main():
    torch.set_float32_matmul_precision('high')
    fold_id = args.fold
    event_name = W.LOOCV_EVENTS[fold_id][0]
    safe = W.safe_event_name(event_name)
    out_dir = f'{PARIS_ROOT}/fold_{fold_id}_{safe}'
    os.makedirs(out_dir, exist_ok=True)
    canon_dir = f'{W.OUT_ROOT}/fold_{fold_id}_{safe}'
    canon_ckpt = f'{canon_dir}/baseline_run/best_model.ckpt'
    assert os.path.exists(canon_ckpt), canon_ckpt

    # ---- dataset + exact canonical split ---------------------------------
    log('loading dataset...')
    base = GatedDatasetLeakFree(W.EMBEDDING_PATH, seq_len=W.SEQ_LEN_OVERRIDE,
                                forecast_horizon=W.HORIZON)
    full_ap = ApEmulatorDataset(base)
    event_to_idx = W.build_event_index(full_ap)
    test_indices = event_to_idx[event_name]
    excl = W.lookback_overlap_indices(full_ap, event_name)
    train_pool = sorted(set(range(len(full_ap))) - excl)
    weights, diag = W.pdf_sampler_weights(full_ap, train_pool)
    np.random.seed(args.seed)
    pool_shuf = list(train_pool); np.random.shuffle(pool_shuf)
    split = int(0.85 * len(pool_shuf))
    train_idx = pool_shuf[:split]; val_idx = pool_shuf[split:]
    pool_to_w = {p: w for p, w in zip(train_pool, weights.tolist())}
    log(f'pool={len(train_pool):,} train={len(train_idx):,} val={len(val_idx):,} '
        f'test={len(test_indices):,} excl={len(excl)}')

    # ---- canonical model + features --------------------------------------
    log('loading canonical fold-20 checkpoint...')
    model = W.load_ckpt(canon_ckpt)
    t0 = time.time()
    Phi_tr, Y_tr, P_tr = extract_phi(model, full_ap, train_idx)
    Phi_va, Y_va, P_va = extract_phi(model, full_ap, val_idx)
    log(f'features: train {tuple(Phi_tr.shape)}  val {tuple(Phi_va.shape)}  '
        f'({time.time()-t0:.0f}s)')

    # Sanity: head reconstruction  Phi @ w + b == model prediction at lead 23
    w = model.dec_proj.weight.detach().cpu().squeeze(0)
    b = float(model.dec_proj.bias.detach().cpu())
    recon = Phi_tr @ w + b
    max_err = float((recon - P_tr).abs().max())
    log(f'head reconstruction max |err| = {max_err:.2e}')
    assert max_err < 1e-4, 'phi/w/b do not reproduce the model head'

    lam_est = estimate_lambda(Phi_tr.numpy().astype(np.float64),
                              (Y_tr.numpy() - b).astype(np.float64),
                              w.numpy().astype(np.float64))
    log(f'closed-form lambda estimate = {lam_est:.4e}  (using override {args.lam:.1e})')

    # ---- PARIS ------------------------------------------------------------
    tr_loader = DataLoader(TensorDataset(Phi_tr, Y_tr), batch_size=4096, shuffle=False)
    va_loader = DataLoader(TensorDataset(Phi_va, Y_va), batch_size=4096, shuffle=False)
    pruner = PARISPruner(phi=lambda x: x, w_NN=w, b_NN=b,
                         prune_fraction=args.prune_fraction, device='cpu',
                         downdate_impl='naive', verbose=True,
                         lambda_override=args.lam)
    t0 = time.time()
    out = pruner.prune(tr_loader, va_loader)
    log(f'PARIS done in {time.time()-t0:.0f}s  K={out["K"]}  lambda={out["lambda"]:.3e}')

    # Map local (position in train_idx) -> global dataset index
    prune_global = [train_idx[k] for k in out['prune_idx']]
    keep_train_idx = [train_idx[k] for k in out['keep_idx']]
    ap72 = full_ap.ap_grid_raw[:, LEAD].numpy()
    pruned_storm = ap72[prune_global] >= AP_STORM_THRESHOLD
    train_storm = ap72[train_idx] >= AP_STORM_THRESHOLD
    storm_share_pruned = float(pruned_storm.mean())
    storm_share_train = float(train_storm.mean())
    log(f'prune diagnostic: storm share pruned={storm_share_pruned*100:.2f}%  '
        f'train pool={storm_share_train*100:.2f}%')
    pd.DataFrame({
        'order': range(len(prune_global)),
        'global_idx': prune_global,
        'ap72_raw': ap72[prune_global],
        'is_storm': pruned_storm.astype(int),
    }).to_csv(f'{out_dir}/paris_prune_indices.csv', index=False)

    # Stop condition 1
    if storm_share_pruned > 0.30:
        log('STOP CONDITION: >30% of pruned samples are storm leads — '
            'method mis-specified for imbalanced target. Not retraining.')
        return
    if args.prune_only:
        log('prune-only mode: done.')
        return

    # ---- retrain on kept subset, identical protocol -----------------------
    weights_kept = torch.tensor([pool_to_w[i] for i in keep_train_idx],
                                dtype=torch.float32)
    log(f'retraining on {len(keep_train_idx):,} kept samples '
        f'(removed {len(prune_global):,})...')
    del Phi_tr, Phi_va, model
    torch.cuda.empty_cache()
    ckpt, train_time = W.retrain(full_ap, keep_train_idx, val_idx, out_dir,
                                 weights_kept, seed=args.seed)
    log(f'retrain done in {train_time:.0f}s  ckpt={ckpt}')
    m2 = W.load_ckpt(ckpt)
    n = W.write_predictions(m2, full_ap, test_indices,
                            f'{out_dir}/baseline_predictions.csv', event_name, fold_id)
    log(f'predictions: {n:,} rows')
    del m2; torch.cuda.empty_cache()

    with open(f'{out_dir}/fold_info.json', 'w') as f:
        json.dump({'fold_id': fold_id, 'event_name': event_name,
                   'variant': 'paris_pruned', 'prune_fraction': args.prune_fraction,
                   'lambda_override': args.lam, 'lambda_estimate': lam_est,
                   'n_pruned': len(prune_global), 'n_kept': len(keep_train_idx),
                   'storm_share_pruned': storm_share_pruned,
                   'storm_share_train': storm_share_train,
                   'canonical_ckpt': canon_ckpt, 'train_time_s': float(train_time),
                   'seed': args.seed}, f, indent=2)

    # ---- score with the canonical scorer ----------------------------------
    import v14_agc_loocv_ensemble as E
    E.CKPT_ROOT = PARIS_ROOT
    log('scoring with v14_agc_loocv_ensemble.run_fold...')
    result, err = E.run_fold(fold_id, full_ap, event_to_idx, DEVICE)
    if err is not None:
        log(f'run_fold error: {err}'); return
    pd.DataFrame(result).to_csv(f'{out_dir}/perscale_paris.csv', index=False)

    canon = pd.read_csv(f'{ROOT}/runs/v14_agc_loocv_ensemble/loocv_perscale_ensemble.csv')
    canon20 = canon[canon.fold == fold_id]
    print('\n' + '=' * 88)
    print(f'FOLD {fold_id} ({event_name})  —  PARIS-pruned vs canonical vs SWPC')
    print('=' * 88)
    for row in result:
        c = canon20[canon20.threshold == row['threshold']].iloc[0]
        print(f"  {row['threshold']}  n_storm={row['n_storm']} pos={row['pos_leads']} "
              f"tau_G1={row['tau_g1']}\n"
              f"      strict: pruned {row['pipeb_strict']:+.3f}  "
              f"canonical {c.pipeb_strict:+.3f}  SWPC {row['swpc_strict']:+.3f}\n"
              f"      tol   : pruned {row['pipeb_tol']:+.3f}  "
              f"canonical {c.pipeb_tol:+.3f}  SWPC {row['swpc_tol']:+.3f}")
        if row['n_storm'] != int(c.n_storm) or row['pos_leads'] != int(c.pos_leads):
            print('      WARNING: n_storm/pos_leads mismatch vs canonical — '
                  'eval misalignment (stop condition 4)')
    print('=' * 88, flush=True)


if __name__ == '__main__':
    main()
