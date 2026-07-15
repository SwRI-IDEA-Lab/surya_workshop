"""Quantify the ap-history timing leak on fold 20 (Gannon).

Preflight: experiments_log.md 2026-07-15 "aplag".

Canonical ap_history's last element is the ap interval BEGINNING at t_0
(not yet elapsed at issue time). This variant lags the aligned ap series
one grid step so the last element is the interval ending at t_0, retrains
fold 20 with the otherwise-identical canonical protocol, and scores with
the fixed-index scorer. Output: runs/v14_agc_ap_emu_aplag/fold_20_Event_22
"""
from __future__ import annotations
import os, sys, json, argparse, time

os.environ.setdefault('EMBEDDING_PATH_OVERRIDE',
    '/media/faraday/andong/Dataspace/GONG_NN/Data/Model_B_AIA_GONG_C3_2011_2025.h5')
os.environ.setdefault('IMG_SLICE_MODE', 'all_three')
os.environ.setdefault('EPOCHS_OVERRIDE', '30')
os.environ.setdefault('CACHE_EMBEDDINGS', '1')

ap = argparse.ArgumentParser()
ap.add_argument('--gpu', type=int, default=7)
ap.add_argument('--fold', type=int, default=20)
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong')
import paris_agc_loocv_fold_v14_ap_emu as W
from dataset_leakfree import GatedDatasetLeakFree
from dataset_leakfree_ap_emu import ApEmulatorDataset
from dataset_leakfree_ap import load_omni2_ap

ROOT = '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong'
LAG_ROOT = f'{ROOT}/runs/v14_agc_ap_emu_aplag'
log = lambda *a: print('[aplag-f20]', *a, flush=True)


def lag_ap_history(ds):
    """Shift the aligned ap series one grid step (last hist element = ap of
    the interval ending at t_0). Returns the original hist for the checks."""
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
    orig = ds.ap_history_raw.numpy().copy()
    # invariants: lagged row is the original row shifted right by one
    assert np.allclose(hist[10, 1:], orig[10, :-1])
    assert np.allclose(hist[10, 0], ap_aligned[9])
    assert np.allclose(hist[5000, 1:], orig[5000, :-1])
    ds.ap_history_raw = torch.from_numpy(hist)
    ds.ap_history = torch.from_numpy(hist / ds.ap_scale)
    log('ap history lagged by one step; invariants OK')


def perday_strict(csv_path, tau_g1=None):
    df = pd.read_csv(csv_path).sort_values(['issue_time', 'lead_h'])
    n = len(df) // 24
    act = df['actual_ap'].values.reshape(n, 24)
    pred = df['pred_ap'].values.reshape(n, 24)
    storm = act.max(axis=1) >= 39
    a_s, p_s = act[storm], pred[storm]
    out = {}
    for day, sl in [('D1', slice(0, 8)), ('D2', slice(8, 16)), ('D3', slice(16, 24))]:
        yt = (a_s[:, sl] >= 39).astype(np.int8)
        yp = (p_s[:, sl] >= (tau_g1 or 26)).astype(np.int8)
        tp = ((yp == 1) & (yt == 1)).sum(); fn = ((yp == 0) & (yt == 1)).sum()
        fp = ((yp == 1) & (yt == 0)).sum(); tn = ((yp == 0) & (yt == 0)).sum()
        out[day] = tp / max(1, tp + fn) - fp / max(1, fp + tn)
    return out


def main():
    torch.set_float32_matmul_precision('high')
    fold_id = args.fold
    event_name = W.LOOCV_EVENTS[fold_id][0]
    safe = W.safe_event_name(event_name)
    out_dir = f'{LAG_ROOT}/fold_{fold_id}_{safe}'
    os.makedirs(out_dir, exist_ok=True)

    log('loading dataset...')
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
    W.write_predictions(m, full_ap, test_indices,
                        f'{out_dir}/baseline_predictions.csv', event_name, fold_id)
    del m; torch.cuda.empty_cache()
    with open(f'{out_dir}/fold_info.json', 'w') as f:
        json.dump({'fold_id': fold_id, 'variant': 'ap_history_lagged_1step',
                   'train_time_s': float(train_time), 'seed': args.seed}, f, indent=2)

    import v14_agc_loocv_ensemble as E
    E.CKPT_ROOT = LAG_ROOT
    log('scoring with fixed-index run_fold...')
    result, err = E.run_fold(fold_id, full_ap, event_to_idx, 'cuda:0')
    if err is not None:
        log(f'run_fold error: {err}'); return
    pd.DataFrame(result).to_csv(f'{out_dir}/perscale_aplag.csv', index=False)

    canon = pd.read_csv(f'{ROOT}/runs/v14_agc_loocv_ensemble/loocv_perscale_ensemble.csv')
    c20 = canon[canon.fold == fold_id]
    print('\n' + '=' * 88)
    print(f'FOLD {fold_id} ({event_name}) — lagged-ap vs canonical (model columns)')
    print('=' * 88)
    for row in result:
        c = c20[c20.threshold == row['threshold']].iloc[0]
        print(f"  {row['threshold']}  tau_G1={row['tau_g1']}  "
              f"strict: lagged {row['pipeb_strict']:+.3f}  canonical {c.pipeb_strict:+.3f}  "
              f"(delta {row['pipeb_strict']-c.pipeb_strict:+.3f})   "
              f"tol: lagged {row['pipeb_tol']:+.3f}  canonical {c.pipeb_tol:+.3f}")
        if row['n_storm'] != int(c.n_storm) or row['pos_leads'] != int(c.pos_leads):
            print('      WARNING: eval alignment mismatch')

    tau_lag = result[0]['tau_g1']
    pd_lag = perday_strict(f'{out_dir}/baseline_predictions.csv', tau_lag)
    pd_can = perday_strict(f'{ROOT}/runs/v14_agc_ap_emu_loocv/fold_{fold_id}_{safe}/'
                           f'baseline_predictions.csv', int(c20.tau_g1.iloc[0]))
    print('\nPer-day G1+ strict (own tau):')
    for d in ['D1', 'D2', 'D3']:
        print(f'  {d}: lagged {pd_lag[d]:+.3f}  canonical {pd_can[d]:+.3f}  '
              f'(delta {pd_lag[d]-pd_can[d]:+.3f})')
    print('=' * 88, flush=True)


if __name__ == '__main__':
    main()
