"""XGB α=0.5 + composite prediction on the temporally cutoff V14-AGC model.

Loads the checkpoint from `runs/v14_agc_cutoff20240831/train/`, retrains
XGB with α=0.5 on the train pool features, and predicts on the OOS test
window (issue_time > 2024-08-31).  Reports G-scale storm-window metrics
vs leak-free SWPC-Ap.
"""
from __future__ import annotations
import os, sys, time, json
os.environ.setdefault('EMBEDDING_PATH_OVERRIDE',
    '/media/faraday/andong/Dataspace/GONG_NN/Data/Model_B_AIA_GONG_C3_2011_2025.h5')
os.environ.setdefault('IMG_SLICE_MODE', 'all_three')
os.environ.setdefault('CACHE_EMBEDDINGS', '1')
os.environ.pop('LOOCV_FILTER', None)

import numpy as np, pandas as pd, torch
sys.path.insert(0, '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong')
from Main_kp_v11_3d_strat import HORIZON, BATCH_SIZE, EMBEDDING_PATH, KP_SCALE_FACTOR
from Main_kp_v14_ap_cls import CLASS_LABELS
from dataset_leakfree import GatedDatasetLeakFree
from dataset_leakfree_ap_emu import ApEmulatorDataset
from dataset_leakfree_ap_residual import kp10_to_ap_lookup
from paris_agc_loocv_fold_v14_xgb_hybrid import HStateExtractorReg, extract_features

RUN_ROOT = '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong/runs/v14_agc_cutoff20240831'
CKPT     = f'{RUN_ROOT}/train/best_model.ckpt'
SPLIT    = f'{RUN_ROOT}/split.npz'
ALPHA    = 0.5
N_CLASSES = 4
CLASS_BOUNDS = [39, 67, 111]  # NOAA G-scale entry points (Kp=5,6,7)
SEQ_LEN  = 56


def ap_to_class(ap):
    return np.searchsorted(CLASS_BOUNDS, ap, side='right')


def main():
    device = 'cuda:0'
    torch.set_float32_matmul_precision('high')
    log = lambda *a: print('[v14agc-cutoff-predict]', *a, flush=True)

    log(f'Loading dataset...')
    base = GatedDatasetLeakFree(EMBEDDING_PATH, seq_len=SEQ_LEN, forecast_horizon=HORIZON)
    ds = ApEmulatorDataset(base)
    ts = pd.DatetimeIndex(base.timestamps)
    log(f'  n={len(ds):,}')

    log(f'Loading train/val/test split from {SPLIT}...')
    sp = np.load(SPLIT)
    train_pool = sp['train_pool'].tolist()
    test_indices = sp['test_indices'].tolist()
    log(f'  train_pool={len(train_pool):,}  test={len(test_indices):,}')

    log(f'Loading V14-AGC cutoff checkpoint from {CKPT}...')
    m = HStateExtractorReg(img_dim=3840, anemo_dim=8, dst_dim=1,
                            hidden_dim=128, num_layers=2, dropout=0.3,
                            forecast_horizon=HORIZON).to(device)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    sd = {k.replace('model.', '', 1): v for k, v in ck['state_dict'].items() if k.startswith('model.')}
    m.load_state_dict(sd, strict=True); m.eval()

    log('Extracting train features...'); t0 = time.time()
    Xtr, ytr, _ = extract_features(m, ds, train_pool, device=device)
    log(f'  Xtr {Xtr.shape}   dist={np.bincount(ytr, minlength=4).tolist()}   ({time.time()-t0:.1f}s)')

    log('Extracting OOS features...'); t0 = time.time()
    Xoos, yoos, meta = extract_features(m, ds, test_indices, device=device)
    log(f'  Xoos {Xoos.shape}   dist={np.bincount(yoos, minlength=4).tolist()}   ({time.time()-t0:.1f}s)')

    del m; torch.cuda.empty_cache()

    log(f'Training XGB with α={ALPHA}...')
    import xgboost as xgb
    counts = np.bincount(ytr, minlength=4).astype(float)
    base_w = counts.sum() / (4 * np.maximum(counts, 1)); base_w /= base_w[0]
    scaled = base_w.copy(); scaled[1:] *= ALPHA
    log(f'  base class_w = {base_w.tolist()}')
    log(f'  scaled       = {scaled.tolist()}')
    sample_w = scaled[ytr]

    clf = xgb.XGBClassifier(
        objective='multi:softprob', num_class=4,
        max_depth=6, learning_rate=0.1, n_estimators=300,
        tree_method='hist', device='cuda:0',
        eval_metric='mlogloss', n_jobs=8, random_state=42,
    )
    t0 = time.time()
    clf.fit(Xtr, ytr, sample_weight=sample_w, verbose=False)
    log(f'  fit {time.time()-t0:.1f}s')

    log('Predicting on OOS window...')
    probs = clf.predict_proba(Xoos)
    pred_class = probs.argmax(axis=1)

    swpc_kp10 = base.swpc_forecasts.numpy() * KP_SCALE_FACTOR
    swpc_ap = kp10_to_ap_lookup(swpc_kp10).reshape(swpc_kp10.shape)

    rows = []
    for j, m_row in enumerate(meta):
        r = dict(sample_idx=m_row['sample_idx'],
                  issue_time=m_row['issue_time'],
                  lead_h=m_row['lead_h'],
                  actual_ap=m_row['actual_ap'],
                  actual_class=m_row['actual_class'],
                  pred_ap_v14=m_row['pred_ap_v14'],
                  pred_class=int(pred_class[j]))
        for c in range(4):
            r[f'prob_{CLASS_LABELS[c]}'] = float(probs[j, c])
        # swpc rows are keyed by ISSUE time = timestamps[sample_idx + seq_len - 1]
        gi = m_row['sample_idx'] + SEQ_LEN - 1
        row = swpc_ap[gi] if gi < len(swpc_ap) else None
        k = (m_row['lead_h'] // 3) - 1
        r['swpc_ap'] = float(row[k]) if row is not None and k < len(row) and np.isfinite(row[k]) else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    df['issue_time'] = pd.to_datetime(df['issue_time'])
    df['day'] = ((df['lead_h'] - 1) // 24) + 1
    df['swpc_class'] = ap_to_class(df['swpc_ap'].fillna(-1).values)
    df.loc[df['swpc_ap'].isna(), 'swpc_class'] = -1
    peak = df.groupby('issue_time')['actual_ap'].max().rename('peak_ap')
    df = df.join(peak, on='issue_time')
    df['is_storm_window'] = df['peak_ap'] >= 39
    df['is_severe_window'] = df['peak_ap'] >= 67

    # Composite + pers2
    probs_arr = df[['prob_quiet', 'prob_G1', 'prob_G2', 'prob_G3+']].values
    storm_probs = probs_arr[:, 1:]
    storm_argmax = storm_probs.argmax(axis=1) + 1
    p_storm_max = storm_probs.max(axis=1)
    argmax_full = probs_arr.argmax(axis=1)
    fire = p_storm_max >= 0.65
    d23_pred = np.where(fire, storm_argmax, 0)
    comp = np.where(df['day'].values == 1, argmax_full, d23_pred)
    df['pred_class_composite'] = comp

    df = df.sort_values(['issue_time', 'lead_h']).reset_index(drop=True)
    flagged = (df['pred_class_composite'].values >= 1).astype(np.int8)
    gated = flagged.copy()
    for _, sub_idx in df.groupby('issue_time').groups.items():
        idx = np.array(list(sub_idx))
        f = flagged[idx]; g = f.copy()
        i = 0
        while i < len(f):
            if f[i] == 0: i += 1; continue
            j = i
            while j < len(f) and f[j] == 1: j += 1
            if j - i < 2: g[i:j] = 0
            i = j
        gated[idx] = g
    df['pred_class_comp_pers2'] = np.where(gated == 1, comp, 0)

    df.to_csv(f'{RUN_ROOT}/oos_predictions.csv', index=False)
    log(f'Saved OOS predictions: {RUN_ROOT}/oos_predictions.csv  ({len(df):,} rows)')

    matched = df[df.swpc_ap.notna()].copy()
    log(f'SWPC coverage on OOS: {len(matched):,}/{len(df):,} ({len(matched)/len(df):.1%})')

    print()
    print('=== OOS (issue_time > 2024-08-31): G-scale storm-window metrics ===')
    print(f'  OOS window: {df.issue_time.min()} to {df.issue_time.max()}')
    peak_per_issue = df.groupby('issue_time')['actual_ap'].max()
    print(f'  n_issues={len(peak_per_issue):,}  n_storm={int((peak_per_issue>=27).sum()):,}  '
          f'n_severe={int((peak_per_issue>=48).sum()):,}  n_G3+={int((peak_per_issue>=80).sum()):,}')

    def metrics(df_slice, col, label):
        storm = df_slice[df_slice.is_storm_window]
        quiet = df_slice[~df_slice.is_storm_window]
        out = []
        for lbl, thr in [('G1+', 1), ('G2+', 2), ('G3+', 3)]:
            yp = (storm[col].values >= thr).astype(int)
            yt = (storm['actual_class'].values >= thr).astype(int)
            tp = int(((yp == 1) & (yt == 1)).sum())
            fn = int(((yp == 0) & (yt == 1)).sum())
            fp = int(((yp == 1) & (yt == 0)).sum())
            tn = int(((yp == 0) & (yt == 0)).sum())
            tpr = tp / max(1, tp + fn); fpr = fp / max(1, fp + tn)
            q = quiet.copy(); q['_fire'] = (q[col].values >= thr).astype(int)
            far = q.groupby('issue_time')['_fire'].sum() * 3
            out.append(dict(config=label, threshold=lbl, tpr=tpr, fpr=fpr, tss=tpr-fpr,
                             tp=tp, fn=fn, fp=fp,
                             fa_h_mean=float(far.mean()) if len(far) else 0,
                             quiet_alert_frac=float((far > 0).mean()) if len(far) else 0))
        return pd.DataFrame(out)

    m_swpc = metrics(matched, 'swpc_class', 'SWPC-Ap')
    m_a05 = metrics(matched, 'pred_class', 'α=0.5 argmax')
    m_a05c = metrics(matched, 'pred_class_comp_pers2', 'α=0.5 comp+pers2')
    tbl = pd.concat([m_swpc, m_a05, m_a05c], ignore_index=True)
    for thr in ['G1+', 'G2+', 'G3+']:
        print(f'\n--- {thr} ---')
        print(tbl[tbl.threshold == thr][['config', 'tpr', 'fpr', 'tss',
                                          'fa_h_mean', 'quiet_alert_frac',
                                          'tp', 'fn', 'fp']].to_string(
            index=False, float_format='%.4f'))
    tbl.to_csv(f'{RUN_ROOT}/oos_metrics.csv', index=False)
    log(f'DONE. Metrics saved to {RUN_ROOT}/oos_metrics.csv')


if __name__ == '__main__':
    main()
