"""Score all 32 lagged-ap-history folds with the canonical (fixed) scorer.

Preflight: experiments_log.md 2026-07-15 "aplag-32fold".
Evaluates each lagged-trained checkpoint ON the lagged dataset
(operational condition) with v14_agc_loocv_ensemble.run_fold — fixed
SWPC indexing, per-fold tau_G1, frozen tau 30/46 at G2+/G3+.

Output: runs/v14_agc_loocv_ensemble/loocv_perscale_aplag.csv
Prints: per-scale medians + CIs (lagged vs canonical vs corrected SWPC)
and paired per-fold deltas (lagged - canonical).
"""
from __future__ import annotations
import os, sys, time, argparse

os.environ.setdefault('EMBEDDING_PATH_OVERRIDE',
    '/media/faraday/andong/Dataspace/GONG_NN/Data/Model_B_AIA_GONG_C3_2011_2025.h5')
os.environ.setdefault('IMG_SLICE_MODE', 'all_three')
os.environ.setdefault('CACHE_EMBEDDINGS', '1')
os.environ.pop('LOOCV_FILTER', None)

ap = argparse.ArgumentParser()
ap.add_argument('--gpu', type=int, default=7)
args = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import numpy as np, pandas as pd, torch
sys.path.insert(0, '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong')
import paris_agc_loocv_fold_v14_ap_emu_pre2015 as P
import paris_agc_loocv_fold_v14_ap_emu as W
import v14_agc_loocv_ensemble as E
from dataset_leakfree import GatedDatasetLeakFree
from dataset_leakfree_ap_emu import ApEmulatorDataset
from dataset_leakfree_ap import load_omni2_ap

ROOT = '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong'
ENS_DIR = f'{ROOT}/runs/v14_agc_loocv_ensemble'
E.LOOCV_EVENTS = W.LOOCV_EVENTS
E.CKPT_ROOT = f'{ROOT}/runs/v14_agc_ap_emu_aplag'
N_BOOT = 1000

log = lambda *a: print('[aplag-eval]', *a, flush=True)


def lag_ap_history(ds):
    """Same one-step lag as the fold worker (inlined; the worker module
    runs a required-args argparse at import time)."""
    omni_ap = load_omni2_ap()
    ap_aligned = omni_ap.reindex(ds.timestamps, method='nearest',
                                 tolerance=pd.Timedelta('1h')).values
    ap_aligned = np.nan_to_num(ap_aligned, nan=0.0).astype(np.float32)
    ap_lag = np.concatenate([ap_aligned[:1], ap_aligned[:-1]])
    n, seq_len = len(ds.base), ds.seq_len
    hist = np.zeros((n, seq_len), dtype=np.float32)
    for idx in range(n):
        end = idx + seq_len
        if end <= len(ap_lag):
            hist[idx, :] = ap_lag[idx:end]
        else:
            k = max(0, len(ap_lag) - idx)
            hist[idx, :k] = ap_lag[idx:idx + k]
    orig = ds.ap_history_raw.numpy()
    assert np.allclose(hist[10, 1:], orig[10, :-1])
    assert np.allclose(hist[5000, 1:], orig[5000, :-1])
    import torch as _t
    ds.ap_history_raw = _t.from_numpy(hist)
    ds.ap_history = _t.from_numpy(hist / ds.ap_scale)
    log('ap history lagged by one step; invariants OK')


def main():
    torch.set_float32_matmul_precision('high')
    log('loading lagged dataset...')
    base = GatedDatasetLeakFree(W.EMBEDDING_PATH, seq_len=56, forecast_horizon=E.HORIZON)
    ds = ApEmulatorDataset(base)
    ds.base = base
    lag_ap_history(ds)
    event_to_idx = E.build_event_index(ds)

    rows = []
    for fold_id in range(len(W.LOOCV_EVENTS)):
        name = W.LOOCV_EVENTS[fold_id][0]
        t0 = time.time()
        result, err = E.run_fold(fold_id, ds, event_to_idx, 'cuda:0')
        if err is not None:
            log(f'fold {fold_id}: SKIP {err}'); continue
        rows.extend(result)
        log(f'fold {fold_id:2d} {name}: '
            + '  '.join(f"{r['threshold']} {r['pipeb_strict']:+.3f}" for r in result)
            + f'  ({time.time()-t0:.0f}s)')
        pd.DataFrame(rows).to_csv(f'{ENS_DIR}/loocv_perscale_aplag.csv', index=False)

    df = pd.DataFrame(rows)
    canon = pd.concat([pd.read_csv(f'{ENS_DIR}/loocv_perscale_ensemble.csv'),
                       pd.read_csv(f'{ENS_DIR}/loocv_perscale_ensemble_pre2015.csv')],
                      ignore_index=True)
    resc = pd.read_csv(f'{ENS_DIR}/loocv_swpc_rescored.csv')
    rng = np.random.default_rng(42)

    def med_ci(v):
        v = pd.Series(v).dropna().values
        if len(v) == 0: return (np.nan,)*3
        b = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(N_BOOT)]
        return np.median(v), np.percentile(b, 2.5), np.percentile(b, 97.5)

    print('\n' + '=' * 96)
    print('32-FOLD LOOCV — LAGGED ap-history vs canonical (both fixed scorer; SWPC corrected)')
    print('=' * 96)
    for lbl in ['G1+', 'G2+', 'G3+']:
        sub = df[df.threshold == lbl]
        csub = canon[canon.threshold == lbl]
        ssub = resc[resc.threshold == lbl]
        for col, src, tag in [('pipeb_strict', sub, 'lagged rule strict   '),
                              ('pipeb_tol', sub, 'lagged rule tol      '),
                              ('pipeb_strict', csub, 'canonical rule strict'),
                              ('pipeb_tol', csub, 'canonical rule tol   '),
                              ('swpc_strict_fixed', ssub, 'SWPC corrected strict')]:
            m, lo, hi = med_ci(src[col])
            print(f'  {lbl} {tag} {m:+.3f} [{lo:+.3f},{hi:+.3f}]')
        merged = sub[['fold', 'pipeb_strict']].merge(
            csub[['fold', 'pipeb_strict']], on='fold', suffixes=('_lag', '_can'))
        d = (merged.pipeb_strict_lag - merged.pipeb_strict_can).dropna().values
        b = [np.median(rng.choice(d, len(d), replace=True)) for _ in range(N_BOOT)]
        print(f'  {lbl} paired lag-canonical  {np.median(d):+.3f} '
              f'[{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}]  (n={len(d)})')
    print(f'\nSaved: {ENS_DIR}/loocv_perscale_aplag.csv', flush=True)


if __name__ == '__main__':
    main()
