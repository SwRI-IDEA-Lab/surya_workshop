"""26-fold LOOCV of the per-scale ensemble under val-selected τ.

For each held-out storm in LOOCV_EVENTS:
  1. Load fold-specific V14-AGC checkpoint (from runs/v14_agc_ap_emu_loocv).
  2. Compute train_pool = all indices not overlapping the held-out storm's lookback.
  3. Split train_pool 85/15 → train + val (seed=42, fold-independent split).
  4. Extract features on train, val, test.
  5. Fit LR on train with balanced-inverse weights.
  6. Sweep τ on val storm-windows for G1+ → val-selected τ.
  7. Apply per-scale ensemble to held-out storm: G1+ Pipeline B, G2+/G3+ LR argmax.
  8. Compute strict + tolerance TSS per G-scale.

Also compute SWPC-Ap for the same fold on the same held-out storm window.

Aggregates: median, IQR, and paired-fold bootstrap CI across folds.
"""
from __future__ import annotations
import os, sys, time, argparse
os.environ.setdefault('EMBEDDING_PATH_OVERRIDE',
    '/media/faraday/andong/Dataspace/GONG_NN/Data/Model_B_AIA_GONG_C3_2011_2025.h5')
os.environ.setdefault('IMG_SLICE_MODE', 'all_three')
os.environ.setdefault('CACHE_EMBEDDINGS', '1')
os.environ.pop('LOOCV_FILTER', None)

import numpy as np, pandas as pd, torch
from scipy.ndimage import maximum_filter1d
sys.path.insert(0, '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong')
from Main_kp_v11_3d_strat import HORIZON, EMBEDDING_PATH, LOOCV_EVENTS, KP_SCALE_FACTOR
from dataset_leakfree import GatedDatasetLeakFree
from dataset_leakfree_ap_emu import ApEmulatorDataset
from dataset_leakfree_ap import AP_SCALE
from dataset_leakfree_ap_residual import kp10_to_ap_lookup
from paris_agc_loocv_fold_v14_xgb_hybrid import HStateExtractorReg
from paris_agc_loocv_fold_v14_xgb_alpha_sweep import extract_features
from paris_agc_loocv_fold_v14_ap_emu import (
    safe_event_name, build_event_index, lookback_overlap_indices,
)

CKPT_ROOT = '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong/runs/v14_agc_ap_emu_loocv'
OUT_DIR = '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong/runs/v14_agc_loocv_ensemble'
os.makedirs(OUT_DIR, exist_ok=True)

NOAA_G = [39, 67, 111]
KERNEL = 9
TAU_SWEEP = list(range(2, 200, 2))
SEED = 42


def tss(yp, yt):
    yp = yp.reshape(-1).astype(np.int8); yt = yt.reshape(-1).astype(np.int8)
    tp = int(((yp == 1) & (yt == 1)).sum()); fn = int(((yp == 0) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum()); tn = int(((yp == 0) & (yt == 0)).sum())
    denom_p = max(1, tp + fn); denom_n = max(1, fp + tn)
    if tp + fn == 0 or fp + tn == 0:
        return np.nan
    return tp / denom_p - fp / denom_n


def tss_pair(yp_2d, yt_2d):
    strict = tss(yp_2d, yt_2d)
    yp_tol = maximum_filter1d(yp_2d, size=KERNEL, axis=1, mode='nearest')
    yt_tol = maximum_filter1d(yt_2d, size=KERNEL, axis=1, mode='nearest')
    tol = tss(yp_tol, yt_tol)
    return strict, tol


def sweep_g1_tau(pred_2d, actual_2d):
    peak = actual_2d.max(axis=1); storm = peak >= NOAA_G[0]
    if storm.sum() == 0:
        return 26  # fallback
    p_s = pred_2d[storm]; a_s = actual_2d[storm]
    yt = (a_s >= NOAA_G[0]).astype(np.int8)
    best = (-999, 26)
    for tau in TAU_SWEEP:
        yp = (p_s >= tau).astype(np.int8)
        s = tss(yp, yt)
        if not np.isnan(s) and s > best[0]:
            best = (s, tau)
    return best[1]


def run_fold(fold_id, ds, event_to_idx, device):
    event_name, e_start, e_end = LOOCV_EVENTS[fold_id]
    safe = safe_event_name(event_name)
    ckpt_path = f'{CKPT_ROOT}/fold_{fold_id}_{safe}/baseline_run/best_model.ckpt'
    if not os.path.exists(ckpt_path):
        return None, f'missing ckpt: {ckpt_path}'

    m = HStateExtractorReg(img_dim=3840, anemo_dim=8, dst_dim=1,
                            hidden_dim=128, num_layers=2, dropout=0.3,
                            forecast_horizon=HORIZON).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {k.replace('model.', '', 1): v for k, v in ck['state_dict'].items() if k.startswith('model.')}
    m.load_state_dict(sd, strict=True); m.eval()

    test_indices = event_to_idx[event_name]
    if not test_indices:
        return None, f'no test indices for {event_name}'
    excl = lookback_overlap_indices(ds, event_name)
    train_pool = sorted(set(range(len(ds))) - excl)
    np.random.seed(SEED)
    pool_shuf = list(train_pool); np.random.shuffle(pool_shuf)
    split_at = int(0.85 * len(pool_shuf))
    train_idx = pool_shuf[:split_at]; val_idx = pool_shuf[split_at:]

    # Extract features
    t0 = time.time()
    Xtr, ytr, _ = extract_features(m, ds, train_idx, device)
    Xva, yva, mva = extract_features(m, ds, val_idx, device)
    Xte, yte, mte = extract_features(m, ds, test_indices, device)
    fx_time = time.time() - t0

    # Fit LR on train pool (train_idx only, not val)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    counts = np.bincount(ytr, minlength=4).astype(float)
    w = counts.sum() / (4 * np.maximum(counts, 1))
    if w[0] == 0: w[0] = 1
    w = w / w[0]
    sample_w = w[ytr]
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)
    lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                             class_weight={i: w[i] for i in range(4)},
                             random_state=SEED)
    t0 = time.time()
    lr.fit(Xtr_s, ytr, sample_weight=sample_w)
    lr_time = time.time() - t0

    # Val τ sweep for G1+
    dfv = pd.DataFrame(mva).sort_values(['issue_time', 'lead_h']).reset_index(drop=True)
    n_va = len(dfv) // HORIZON
    pred_va_win = dfv['pred_ap_v14'].values.reshape(n_va, HORIZON)
    act_va_win = dfv['actual_ap'].values.reshape(n_va, HORIZON)
    tau_g1 = sweep_g1_tau(pred_va_win, act_va_win)

    # Apply to held-out storm
    dfte = pd.DataFrame(mte).sort_values(['issue_time', 'lead_h']).reset_index(drop=True)
    lr_argmax_te = lr.predict_proba(Xte_s).argmax(axis=1)
    dfte_source = pd.DataFrame(mte)
    dfte_source['lr_argmax'] = lr_argmax_te
    dfte = dfte_source.sort_values(['issue_time', 'lead_h']).reset_index(drop=True)
    n_te = len(dfte) // HORIZON
    if n_te == 0:
        return None, f'no test windows'

    pred_te_win = dfte['pred_ap_v14'].values.reshape(n_te, HORIZON)
    act_te_win = dfte['actual_ap'].values.reshape(n_te, HORIZON)
    actual_cls_te_win = dfte['actual_class'].values.reshape(n_te, HORIZON)
    lr_te_win = dfte['lr_argmax'].values.reshape(n_te, HORIZON)

    peak = act_te_win.max(axis=1); storm = peak >= NOAA_G[0]
    if storm.sum() == 0:
        return None, f'no storm-window issues for {event_name}'

    p_s = pred_te_win[storm]
    a_s = act_te_win[storm]
    lr_s = lr_te_win[storm]

    # SWPC-Ap on same fold
    swpc_kp10 = ds.base.swpc_forecasts.numpy() * KP_SCALE_FACTOR
    swpc_ap_full = kp10_to_ap_lookup(swpc_kp10).reshape(swpc_kp10.shape)
    swpc_te = np.zeros_like(pred_te_win)
    for i, idx in enumerate(test_indices):
        # swpc_forecasts rows are keyed by ISSUE time; sample idx issues at
        # timestamps[idx + seq_len - 1] (same convention as __getitem__).
        issue_row = idx + ds.seq_len - 1
        if issue_row < len(swpc_ap_full):
            swpc_te[i] = swpc_ap_full[issue_row]
        else:
            swpc_te[i] = np.nan
    swpc_s = swpc_te[storm]

    # Compute TSS
    rows = []
    for lbl, gth in [('G1+', 0), ('G2+', 1), ('G3+', 2)]:
        thr = NOAA_G[gth]
        yt = (a_s >= thr).astype(np.int8)
        # SWPC
        yp_swpc = (swpc_s >= thr).astype(np.int8)
        s_swpc, t_swpc = tss_pair(yp_swpc, yt)
        # Ensemble
        if lbl == 'G1+':
            yp_ens = (p_s >= tau_g1).astype(np.int8)
        elif lbl == 'G2+':
            yp_ens = (lr_s >= 2).astype(np.int8)
        else:
            yp_ens = (lr_s >= 3).astype(np.int8)
        s_ens, t_ens = tss_pair(yp_ens, yt)
        # Pipeline B baseline (val-τ for G1+ only; for G2+/G3+ we use best-guess proxies 30/46 nT)
        tau_g2_default, tau_g3_default = 30, 46
        if lbl == 'G1+':
            yp_pb = (p_s >= tau_g1).astype(np.int8)
        elif lbl == 'G2+':
            yp_pb = (p_s >= tau_g2_default).astype(np.int8)
        else:
            yp_pb = (p_s >= tau_g3_default).astype(np.int8)
        s_pb, t_pb = tss_pair(yp_pb, yt)
        rows.append(dict(fold=fold_id, event=event_name, threshold=lbl,
                          swpc_strict=s_swpc, swpc_tol=t_swpc,
                          pipeb_strict=s_pb, pipeb_tol=t_pb,
                          ens_strict=s_ens, ens_tol=t_ens,
                          n_storm=int(storm.sum()),
                          pos_leads=int(yt.sum()),
                          tau_g1=tau_g1,
                          fx_time=fx_time, lr_time=lr_time))
    return rows, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=str, default='0-25',
                          help='fold range, e.g., "0-25" or "0,5,10"')
    args = parser.parse_args()
    if '-' in args.folds:
        a, b = map(int, args.folds.split('-'))
        fold_list = list(range(a, b + 1))
    else:
        fold_list = [int(x) for x in args.folds.split(',')]

    device = 'cuda:0'
    torch.set_float32_matmul_precision('high')
    log = lambda *a: print('[loocv]', *a, flush=True)

    log(f'Folds to run: {fold_list}')
    log('Loading dataset (once)...')
    base = GatedDatasetLeakFree(EMBEDDING_PATH, seq_len=56, forecast_horizon=HORIZON)
    ds = ApEmulatorDataset(base)
    log(f'  n={len(ds):,}')
    # Attach base for SWPC access
    ds.base = base

    event_to_idx = build_event_index(ds)
    log(f'  event coverage: {sum(1 for v in event_to_idx.values() if v)}/{len(LOOCV_EVENTS)}')

    all_rows = []
    for fold_id in fold_list:
        event_name, _, _ = LOOCV_EVENTS[fold_id]
        log(f'\n=== FOLD {fold_id}: {event_name} ===')
        t0 = time.time()
        result, err = run_fold(fold_id, ds, event_to_idx, device)
        if err is not None:
            log(f'  SKIP: {err}')
            continue
        for row in result:
            log(f'  {row["threshold"]}  '
                f'n_storm={row["n_storm"]}  pos={row["pos_leads"]}  τ_G1={row["tau_g1"]}  '
                f'SWPC strict={row["swpc_strict"]:+.3f}  '
                f'PipeB strict={row["pipeb_strict"]:+.3f}  '
                f'Ens strict={row["ens_strict"]:+.3f}  '
                f'Ens tol={row["ens_tol"]:+.3f}')
        all_rows.extend(result)
        log(f'  fold time: {time.time() - t0:.1f}s')

        # Save incrementally
        pd.DataFrame(all_rows).to_csv(f'{OUT_DIR}/loocv_perscale_ensemble.csv', index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(f'{OUT_DIR}/loocv_perscale_ensemble.csv', index=False)
    log(f'\nSaved {len(df)} rows to {OUT_DIR}/loocv_perscale_ensemble.csv')

    # Aggregate summary
    print()
    print('=' * 90)
    print('LOOCV PER-SCALE ENSEMBLE — 26-FOLD SUMMARY')
    print('=' * 90)
    for lbl in ['G1+', 'G2+', 'G3+']:
        sub = df[df.threshold == lbl].dropna(subset=['ens_strict'])
        if len(sub) == 0:
            continue
        print(f'\n--- {lbl} (n_folds={len(sub)}) ---')
        for method in ['swpc', 'pipeb', 'ens']:
            for metric in ['strict', 'tol']:
                col = f'{method}_{metric}'
                vals = sub[col].dropna().values
                if len(vals) == 0:
                    continue
                # Fold bootstrap CI
                rng = np.random.default_rng(42)
                boots = [np.median(rng.choice(vals, len(vals), replace=True))
                          for _ in range(1000)]
                print(f'  {method:<5} {metric:<7} '
                      f'median {np.median(vals):+.4f}  '
                      f'IQR [{np.percentile(vals, 25):+.3f}, {np.percentile(vals, 75):+.3f}]  '
                      f'boot-median CI [{np.percentile(boots, 2.5):+.3f}, '
                      f'{np.percentile(boots, 97.5):+.3f}]')


if __name__ == '__main__':
    main()
