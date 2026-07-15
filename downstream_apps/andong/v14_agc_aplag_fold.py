"""Per-fold worker: canonical V14-AGC protocol with lagged ap history.

Preflight: experiments_log.md 2026-07-15 "aplag-32fold".
Same as the canonical fold worker except ap_aligned is shifted one grid
step (last history element = the 3-h interval ENDING at t_0). 32-event
catalog. Training + predictions only; scoring is a separate aggregate
pass (v14_agc_aplag_eval.py).
"""
from __future__ import annotations
import os, sys, json, argparse

os.environ.setdefault('EMBEDDING_PATH_OVERRIDE',
    '/media/faraday/andong/Dataspace/GONG_NN/Data/Model_B_AIA_GONG_C3_2011_2025.h5')
os.environ.setdefault('IMG_SLICE_MODE', 'all_three')
os.environ.setdefault('EPOCHS_OVERRIDE', '30')
os.environ.setdefault('CACHE_EMBEDDINGS', '1')

ap = argparse.ArgumentParser()
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--gpu', type=int, required=True)
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong')
import paris_agc_loocv_fold_v14_ap_emu_pre2015 as P   # extends catalog to 32 events
import paris_agc_loocv_fold_v14_ap_emu as W
from dataset_leakfree import GatedDatasetLeakFree
from dataset_leakfree_ap_emu import ApEmulatorDataset
from dataset_leakfree_ap import load_omni2_ap

LAG_ROOT = ('/media/faraday/andong/Workspace/surya_workshop/downstream_apps/'
            'andong/runs/v14_agc_ap_emu_aplag')
log = lambda *a: print(f'[aplag f{args.fold}]', *a, flush=True)


def lag_ap_history(ds):
    """Shift the aligned ap series one grid step (last hist element = ap of
    the interval ending at t_0). Same function as v14_agc_aplag_fold20
    (inlined: that module runs argparse at import time)."""
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
    assert np.allclose(hist[10, 0], ap_aligned[9])
    assert np.allclose(hist[5000, 1:], orig[5000, :-1])
    ds.ap_history_raw = torch.from_numpy(hist)
    ds.ap_history = torch.from_numpy(hist / ds.ap_scale)
    log('ap history lagged by one step; invariants OK')


def main():
    torch.set_float32_matmul_precision('high')
    fold_id = args.fold
    event_name = W.LOOCV_EVENTS[fold_id][0]
    safe = W.safe_event_name(event_name)
    out_dir = f'{LAG_ROOT}/fold_{fold_id}_{safe}'
    if os.path.exists(f'{out_dir}/baseline_predictions.csv'):
        log('predictions exist; skipping'); return
    os.makedirs(out_dir, exist_ok=True)

    base = GatedDatasetLeakFree(W.EMBEDDING_PATH, seq_len=W.SEQ_LEN_OVERRIDE,
                                forecast_horizon=W.HORIZON)
    full_ap = ApEmulatorDataset(base)
    lag_ap_history(full_ap)

    event_to_idx = W.build_event_index(full_ap)
    test_indices = event_to_idx[event_name]
    excl = W.lookback_overlap_indices(full_ap, event_name)
    train_pool = sorted(set(range(len(full_ap))) - excl)
    weights, _ = W.pdf_sampler_weights(full_ap, train_pool)
    np.random.seed(args.seed)
    pool_shuf = list(train_pool); np.random.shuffle(pool_shuf)
    split = int(0.85 * len(pool_shuf))
    train_idx = pool_shuf[:split]; val_idx = pool_shuf[split:]
    pool_to_w = {p: w for p, w in zip(train_pool, weights.tolist())}
    weights_train = torch.tensor([pool_to_w[i] for i in train_idx], dtype=torch.float32)
    log(f'train={len(train_idx)} val={len(val_idx)} test={len(test_indices)}')

    ckpt, train_time = W.retrain(full_ap, train_idx, val_idx, out_dir,
                                 weights_train, seed=args.seed)
    log(f'retrain done in {train_time:.0f}s')
    m = W.load_ckpt(ckpt)
    n = W.write_predictions(m, full_ap, test_indices,
                            f'{out_dir}/baseline_predictions.csv', event_name, fold_id)
    with open(f'{out_dir}/fold_info.json', 'w') as f:
        json.dump({'fold_id': fold_id, 'event_name': event_name,
                   'variant': 'ap_history_lagged_1step',
                   'train_time_s': float(train_time), 'seed': args.seed}, f, indent=2)
    log(f'DONE ({n:,} prediction rows)')


if __name__ == '__main__':
    main()
