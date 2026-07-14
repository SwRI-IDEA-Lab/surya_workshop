"""Evaluate the 6 pre-2015 LOOCV folds (Events #28-#33) with the exact
canonical scorer.

Reuses v14_agc_loocv_ensemble.run_fold unchanged (same tss/tss_pair code,
NOAA bounds [39,67,111], tolerance kernel 9, leak-free SWPC-Ap, LR ensemble,
val-selected G1+ tau) — only the catalog is extended and CKPT_ROOT points at
the pre-2015 output root.  Also adds ap-persistence and 27-day-recurrence
rows using the same logic as v14_agc_loocv_trivial_baselines.py.

Output: runs/v14_agc_loocv_ensemble/loocv_perscale_ensemble_pre2015.csv
        runs/v14_agc_loocv_ensemble/loocv_trivial_baselines_pre2015.csv
(canonical 26-fold CSVs untouched)
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong')

import paris_agc_loocv_fold_v14_ap_emu_pre2015 as P   # patches W.LOOCV_EVENTS
import paris_agc_loocv_fold_v14_ap_emu as W
import v14_agc_loocv_ensemble as E

import numpy as np
import pandas as pd
import torch

E.LOOCV_EVENTS = W.LOOCV_EVENTS
E.CKPT_ROOT = ('/media/faraday/andong/Workspace/surya_workshop/downstream_apps/'
               'andong/runs/v14_agc_ap_emu_loocv_pre2015')
OUT_DIR = ('/media/faraday/andong/Workspace/surya_workshop/downstream_apps/'
           'andong/runs/v14_agc_loocv_ensemble')
NEW_FOLDS = list(range(P.FIRST_NEW_FOLD, len(W.LOOCV_EVENTS)))   # 26..31
CANON_CSV = f'{OUT_DIR}/loocv_perscale_ensemble.csv'


def trivial_rows(ds, event_to_idx):
    """persistence + 27-day recurrence for the new folds, canonical protocol."""
    from dataset_leakfree_ap import load_omni2_ap
    ts_all = pd.DatetimeIndex(ds.timestamps)
    omni_ap = load_omni2_ap()
    ap_series = pd.Series(
        omni_ap.reindex(ts_all, method='nearest', tolerance=pd.Timedelta('1h')).values,
        index=ts_all)
    act_grid = np.asarray(ds.ap_grid_raw)
    pers_src = np.asarray(ds.ap_history_raw)[:, -1]
    offset = ds.seq_len - 1
    rows = []
    for fold_id in NEW_FOLDS:
        name = W.LOOCV_EVENTS[fold_id][0]
        idxs = event_to_idx[name]
        act = act_grid[idxs]                                   # (n, 24)
        pers = np.repeat(pers_src[idxs][:, None], E.HORIZON, axis=1)
        rec = np.full_like(act, np.nan, dtype=np.float64)
        for i, idx in enumerate(idxs):
            t0 = ts_all[idx + offset]
            for k in range(E.HORIZON):
                tt = t0 + pd.Timedelta(hours=3 * (k + 1)) - pd.Timedelta(days=27)
                try:
                    v = ap_series.asof(tt)
                except Exception:
                    v = np.nan
                rec[i, k] = v
        peak = act.max(axis=1); storm = peak >= E.NOAA_G[0]
        if storm.sum() == 0:
            continue
        a_s = act[storm]
        for lbl, thr in zip(['G1+', 'G2+', 'G3+'], E.NOAA_G):
            yt = (a_s >= thr).astype(np.int8)
            sp, tp_ = E.tss_pair((pers[storm] >= thr).astype(np.int8), yt)
            r = np.nan_to_num(rec[storm], nan=0.0)
            sr, tr = E.tss_pair((r >= thr).astype(np.int8), yt)
            rows.append(dict(fold=fold_id, event=name, threshold=lbl,
                             pers_strict=sp, pers_tol=tp_,
                             rec_strict=sr, rec_tol=tr,
                             n_storm=int(storm.sum()), pos_leads=int(yt.sum())))
    return rows


def main():
    device = 'cuda:0'
    torch.set_float32_matmul_precision('high')
    log = lambda *a: print('[pre2015-eval]', *a, flush=True)

    from dataset_leakfree import GatedDatasetLeakFree
    from dataset_leakfree_ap_emu import ApEmulatorDataset
    log('Loading dataset (once)...')
    base = GatedDatasetLeakFree(E.EMBEDDING_PATH, seq_len=56, forecast_horizon=E.HORIZON)
    ds = ApEmulatorDataset(base)
    ds.base = base
    event_to_idx = E.build_event_index(ds)

    all_rows = []
    for fold_id in NEW_FOLDS:
        name = W.LOOCV_EVENTS[fold_id][0]
        log(f'=== FOLD {fold_id}: {name} ===')
        t0 = time.time()
        result, err = E.run_fold(fold_id, ds, event_to_idx, device)
        if err is not None:
            log(f'  SKIP: {err}'); continue
        for row in result:
            log(f"  {row['threshold']}  n_storm={row['n_storm']} pos={row['pos_leads']} "
                f"tau_G1={row['tau_g1']}  SWPC strict={row['swpc_strict']:+.3f}  "
                f"rule strict={row['pipeb_strict']:+.3f} tol={row['pipeb_tol']:+.3f}  "
                f"ens strict={row['ens_strict']:+.3f}")
        all_rows.extend(result)
        log(f'  fold time {time.time()-t0:.0f}s')
        pd.DataFrame(all_rows).to_csv(
            f'{OUT_DIR}/loocv_perscale_ensemble_pre2015.csv', index=False)

    log('trivial baselines...')
    triv = trivial_rows(ds, event_to_idx)
    pd.DataFrame(triv).to_csv(
        f'{OUT_DIR}/loocv_trivial_baselines_pre2015.csv', index=False)

    # ---- summaries -------------------------------------------------------
    df = pd.DataFrame(all_rows)
    dft = pd.DataFrame(triv)
    canon = pd.read_csv(CANON_CSV)
    rng = np.random.default_rng(42)

    def med_ci(v):
        v = pd.Series(v).dropna().values
        if len(v) == 0: return (np.nan,)*3
        b = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(1000)]
        return np.median(v), np.percentile(b, 2.5), np.percentile(b, 97.5)

    print('\n' + '='*100)
    print('PRE-2015 EXTENSION — 6 NEW FOLDS (Events #28-#33), frozen protocol')
    print('='*100)
    for lbl in ['G1+', 'G2+', 'G3+']:
        sub = df[df.threshold == lbl]; subt = dft[dft.threshold == lbl]
        print(f'\n--- {lbl} (n_folds={len(sub)}) ---')
        for col, src, tag in [('pers_strict', subt, 'persistence'),
                              ('rec_strict', subt, 'recurrence '),
                              ('swpc_strict', sub, 'SWPC-Ap    '),
                              ('pipeb_strict', sub, 'thresh rule'),
                              ('ens_strict', sub, 'ensemble   ')]:
            m, lo, hi = med_ci(src[col])
            print(f'  {tag} strict median {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]')
        m, lo, hi = med_ci(sub['pipeb_tol'])
        print(f'  thresh rule tol    median {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]')
        # combined 32-fold reference
        comb = pd.concat([canon[canon.threshold == lbl], sub])
        mc, loc_, hic = med_ci(comb['pipeb_strict'])
        ms, _, _ = med_ci(comb['swpc_strict'])
        print(f'  COMBINED 32-fold: rule strict {mc:+.3f} [{loc_:+.3f},{hic:+.3f}]  SWPC {ms:+.3f}')

    # stop-condition report
    print('\n--- stop-condition checks ---')
    g1 = df[df.threshold == 'G1+']
    adv = (g1.pipeb_strict - g1.swpc_strict).values
    print(f'  G1+ advantage per fold: {np.round(adv, 3)}')
    print(f'  median adv {np.median(adv):+.3f}; folds negative: {(adv < 0).sum()}/6')
    drop_one = [np.median(np.delete(adv, i)) for i in range(len(adv))]
    print(f'  drop-one medians: {np.round(drop_one, 3)}  (sign-stable: {all(d > 0 for d in drop_one) or all(d < 0 for d in drop_one)})')
    if (g1.swpc_strict > 0.5).any():
        print('  WARNING: SWPC strict TSS > +0.5 on a fold — check old-era keying')


if __name__ == '__main__':
    main()
