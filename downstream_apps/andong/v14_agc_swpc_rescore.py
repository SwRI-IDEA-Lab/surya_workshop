"""Rescore the SWPC-Ap baseline after the issue-row indexing fix.

Preflight: experiments_log.md 2026-07-15 "SWPC-rescore".

The V14-AGC-era scorers keyed base.swpc_forecasts by raw sample index; the
tensor is keyed by ISSUE time, i.e. row idx + seq_len - 1 (the dataset's own
__getitem__ convention). Model columns are untouched everywhere; this script
recomputes only the SWPC columns:

  A. 32-fold LOOCV (canonical 26 + pre-2015 6):
     runs/v14_agc_loocv_ensemble/loocv_swpc_rescored.csv
  B. OOS forward test: runs/v14_agc_cutoff20240831/oos_predictions_swpcfix.csv
     + aggregate and per-day tables recomputed.

Alignment gates (script aborts if any fails):
  1. swpc_forecasts[idx+seq_len-1] == __getitem__(idx) y_swpc (100 samples)
  2. OLD buggy index inside this harness reproduces the canonical CSV swpc
     columns (proves the harness is aligned; delta is then the fix alone)
  3. corrected SWPC during Gannon reaches G3+ range at short leads
"""
from __future__ import annotations
import os, sys

os.environ.setdefault('EMBEDDING_PATH_OVERRIDE',
    '/media/faraday/andong/Dataspace/GONG_NN/Data/Model_B_AIA_GONG_C3_2011_2025.h5')
os.environ.setdefault('IMG_SLICE_MODE', 'all_three')
os.environ.pop('LOOCV_FILTER', None)

import numpy as np, pandas as pd, torch
from scipy.ndimage import maximum_filter1d
sys.path.insert(0, '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong')
import paris_agc_loocv_fold_v14_ap_emu_pre2015 as P   # extends W.LOOCV_EVENTS to 32
import paris_agc_loocv_fold_v14_ap_emu as W
from Main_kp_v11_3d_strat import HORIZON, EMBEDDING_PATH, KP_SCALE_FACTOR
from dataset_leakfree import GatedDatasetLeakFree
from dataset_leakfree_ap_emu import ApEmulatorDataset
from dataset_leakfree_ap_residual import kp10_to_ap_lookup

ROOT = '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong'
ENS_DIR = f'{ROOT}/runs/v14_agc_loocv_ensemble'
OOS_ROOT = f'{ROOT}/runs/v14_agc_cutoff20240831'
NOAA_G = [39, 67, 111]
KERNEL = 9          # pooled-lead tolerance (canonical)
KERNEL_D = 5        # within-day tolerance (per-day convention)
DAY_SLICES = {'D1': slice(0, 8), 'D2': slice(8, 16), 'D3': slice(16, 24)}
TAU = {'G1+': 26, 'G2+': 30, 'G3+': 46}
N_BOOT = 1000

log = lambda *a: print('[swpc-rescore]', *a, flush=True)


def tss(yp, yt):
    yp = np.asarray(yp).reshape(-1).astype(np.int8)
    yt = np.asarray(yt).reshape(-1).astype(np.int8)
    tp = int(((yp == 1) & (yt == 1)).sum()); fn = int(((yp == 0) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum()); tn = int(((yp == 0) & (yt == 0)).sum())
    if tp + fn == 0 or fp + tn == 0:
        return np.nan
    return tp / (tp + fn) - fp / (fp + tn)


def tss_pair(yp_2d, yt_2d, kernel=KERNEL):
    strict = tss(yp_2d, yt_2d)
    yp_tol = maximum_filter1d(yp_2d, size=kernel, axis=1, mode='nearest')
    yt_tol = maximum_filter1d(yt_2d, size=kernel, axis=1, mode='nearest')
    return strict, tss(yp_tol, yt_tol)


def med_ci(v, rng):
    v = pd.Series(v).dropna().values
    if len(v) == 0: return (np.nan,) * 3
    b = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(N_BOOT)]
    return np.median(v), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    log('Loading dataset (no embedding cache needed)...')
    base = GatedDatasetLeakFree(EMBEDDING_PATH, seq_len=56, forecast_horizon=HORIZON)
    ds = ApEmulatorDataset(base)
    event_to_idx = W.build_event_index(ds)
    seq_off = ds.seq_len - 1

    swpc_kp10 = base.swpc_forecasts.numpy() * KP_SCALE_FACTOR
    swpc_ap_full = kp10_to_ap_lookup(swpc_kp10).reshape(swpc_kp10.shape)
    swpc_valid = base.swpc_valid.numpy()
    act_full = np.asarray(ds.ap_grid_raw)

    # ---- gate 1: issue-row equivalence with __getitem__ -------------------
    log('gate 1: __getitem__ equivalence (100 random samples)...')
    rng0 = np.random.default_rng(0)
    for idx in rng0.integers(0, len(ds) - 1, 100):
        y_swpc_item = base[int(idx)][4].numpy()          # scaled Kp
        row = base.swpc_forecasts[int(idx) + seq_off].numpy()
        assert np.allclose(y_swpc_item, row, atol=1e-6), f'gate 1 FAILED at idx={idx}'
    log('  gate 1 PASSED')

    # ---- A. LOOCV rescore ---------------------------------------------------
    canon = pd.concat([pd.read_csv(f'{ENS_DIR}/loocv_perscale_ensemble.csv'),
                       pd.read_csv(f'{ENS_DIR}/loocv_perscale_ensemble_pre2015.csv')],
                      ignore_index=True)
    rows, gate2_fail = [], []
    for fold_id in range(len(W.LOOCV_EVENTS)):
        name = W.LOOCV_EVENTS[fold_id][0]
        idxs = event_to_idx[name]
        if not idxs: continue
        act = act_full[idxs]
        storm = act.max(axis=1) >= NOAA_G[0]
        if storm.sum() == 0: continue
        a_s = act[storm]
        idxs_arr = np.array(idxs)
        stale = swpc_ap_full[idxs_arr]                    # OLD buggy keying
        fixed = swpc_ap_full[idxs_arr + seq_off]          # corrected keying
        vmask = swpc_valid[idxs_arr + seq_off]
        st_s, fx_s, vm_s = stale[storm], fixed[storm], vmask[storm]
        for lbl, thr in zip(['G1+', 'G2+', 'G3+'], NOAA_G):
            yt = (a_s >= thr).astype(np.int8)
            s_stale, t_stale = tss_pair((st_s >= thr).astype(np.int8), yt)
            s_fix, t_fix = tss_pair((fx_s >= thr).astype(np.int8), yt)
            # coverage-masked strict (sensitivity): drop uncovered bins
            yp_m = (fx_s >= thr).astype(np.int8)[vm_s]
            yt_m = yt[vm_s]
            s_fix_cov = tss(yp_m, yt_m)
            c = canon[(canon.fold == fold_id) & (canon.threshold == lbl)]
            csv_swpc = float(c.swpc_strict.iloc[0]) if len(c) else np.nan
            if len(c) and abs(s_stale - csv_swpc) > 0.005:
                gate2_fail.append((fold_id, lbl, s_stale, csv_swpc))
            rows.append(dict(fold=fold_id, event=name, threshold=lbl,
                             swpc_strict_stale=s_stale, swpc_tol_stale=t_stale,
                             swpc_strict_fixed=s_fix, swpc_tol_fixed=t_fix,
                             swpc_strict_fixed_covmask=s_fix_cov,
                             coverage=float(vm_s.mean()),
                             csv_swpc_strict=csv_swpc,
                             n_storm=int(storm.sum()), pos_leads=int(yt.sum())))
    df = pd.DataFrame(rows)

    log('gate 2: stale-index reproduction vs canonical CSVs...')
    if gate2_fail:
        for f in gate2_fail[:10]: log('  FAIL', f)
        raise SystemExit('gate 2 FAILED — harness misaligned, numbers void')
    log(f'  gate 2 PASSED (all {len(df)} cells within 0.005 of CSV swpc_strict)')

    # ---- gate 3: Gannon physical spot check --------------------------------
    g_idx = np.array(event_to_idx['Event #22'])
    g_fixed = swpc_ap_full[g_idx + seq_off]
    g_stale = swpc_ap_full[g_idx]
    log(f'gate 3: Gannon corrected SWPC max={g_fixed.max():.0f} nT '
        f'(stale-index max={g_stale.max():.0f}); expect G3+ range (>=111)')
    assert g_fixed.max() >= NOAA_G[2], 'gate 3 FAILED — corrected SWPC still quiet in Gannon'
    log('  gate 3 PASSED')

    df.to_csv(f'{ENS_DIR}/loocv_swpc_rescored.csv', index=False)

    rng = np.random.default_rng(42)
    print('\n' + '=' * 100)
    print('32-FOLD LOOCV — corrected SWPC-Ap vs model (model columns from canonical CSVs, unchanged)')
    print('=' * 100)
    for lbl in ['G1+', 'G2+', 'G3+']:
        sub = df[df.threshold == lbl]
        csub = canon[canon.threshold == lbl]
        m_st, lo_st, hi_st = med_ci(sub.swpc_strict_stale, rng)
        m_fx, lo_fx, hi_fx = med_ci(sub.swpc_strict_fixed, rng)
        m_ft, lo_ft, hi_ft = med_ci(sub.swpc_tol_fixed, rng)
        m_cv, _, _ = med_ci(sub.swpc_strict_fixed_covmask, rng)
        m_pb, lo_pb, hi_pb = med_ci(csub.pipeb_strict, rng)
        m_pt, lo_pt, hi_pt = med_ci(csub.pipeb_tol, rng)
        m_en, lo_en, hi_en = med_ci(csub.ens_strict, rng)
        print(f'\n--- {lbl} (n_folds={len(sub)}) ---')
        print(f'  SWPC stale (as published) strict {m_st:+.3f} [{lo_st:+.3f},{hi_st:+.3f}]')
        print(f'  SWPC FIXED               strict {m_fx:+.3f} [{lo_fx:+.3f},{hi_fx:+.3f}]   tol {m_ft:+.3f} [{lo_ft:+.3f},{hi_ft:+.3f}]')
        print(f'  SWPC FIXED cov-masked    strict {m_cv:+.3f}')
        print(f'  rule (unchanged)         strict {m_pb:+.3f} [{lo_pb:+.3f},{hi_pb:+.3f}]   tol {m_pt:+.3f} [{lo_pt:+.3f},{hi_pt:+.3f}]')
        print(f'  ensemble (unchanged)     strict {m_en:+.3f} [{lo_en:+.3f},{hi_en:+.3f}]')
        # paired per-fold delta rule - fixed SWPC
        merged = sub.merge(csub[['fold', 'pipeb_strict', 'pipeb_tol']], on='fold')
        d = (merged.pipeb_strict - merged.swpc_strict_fixed).values
        b = [np.median(rng.choice(d, len(d), replace=True)) for _ in range(N_BOOT)]
        print(f'  paired rule-SWPCfixed    strict {np.median(d):+.3f} [{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}]')

    # ---- B. OOS rescore -----------------------------------------------------
    log('\nOOS rescore...')
    oos = pd.read_csv(f'{OOS_ROOT}/oos_predictions.csv')
    gi = oos['sample_idx'].values + seq_off
    k = (oos['lead_h'].values // 3) - 1
    ok = gi < len(swpc_ap_full)
    new_swpc = np.full(len(oos), np.nan)
    new_swpc[ok] = swpc_ap_full[gi[ok], k[ok]]
    new_valid = np.zeros(len(oos), bool)
    new_valid[ok] = swpc_valid[gi[ok], k[ok]]
    oos['swpc_ap_stale'] = oos['swpc_ap']
    oos['swpc_ap'] = new_swpc
    oos['swpc_valid'] = new_valid
    oos.to_csv(f'{OOS_ROOT}/oos_predictions_swpcfix.csv', index=False)

    oos = oos.sort_values(['issue_time', 'lead_h']).reset_index(drop=True)
    n = len(oos) // HORIZON
    act = oos['actual_ap'].values.reshape(n, HORIZON)
    pred = oos['pred_ap_v14'].values.reshape(n, HORIZON)
    swf = oos['swpc_ap'].values.reshape(n, HORIZON)
    sws = oos['swpc_ap_stale'].values.reshape(n, HORIZON)
    storm = act.max(axis=1) >= NOAA_G[0]
    a_s, p_s = act[storm], pred[storm]
    f_s, st_s = np.nan_to_num(swf[storm]), np.nan_to_num(sws[storm])
    log(f'OOS storm issues = {int(storm.sum())} (expect 629)')

    print('\n' + '=' * 100)
    print('OOS FORWARD TEST (2024-09 -> 2025-04) — corrected SWPC, rule unchanged (tau 26/30/46)')
    print('=' * 100)
    oos_rows = []
    for lbl, thr in zip(['G1+', 'G2+', 'G3+'], NOAA_G):
        yt = (a_s >= thr).astype(np.int8)
        s_r, t_r = tss_pair((p_s >= TAU[lbl]).astype(np.int8), yt)
        s_f, t_f = tss_pair((f_s >= thr).astype(np.int8), yt)
        s_o, t_o = tss_pair((st_s >= thr).astype(np.int8), yt)
        print(f'  {lbl}: rule strict {s_r:+.3f} tol {t_r:+.3f} | SWPC FIXED strict {s_f:+.3f} tol {t_f:+.3f} '
              f'| SWPC stale strict {s_o:+.3f} (published)')
        oos_rows.append(dict(threshold=lbl, scope='pooled', rule_strict=s_r, rule_tol=t_r,
                             swpc_fixed_strict=s_f, swpc_fixed_tol=t_f,
                             swpc_stale_strict=s_o))

    print('\nPER-DAY (within-day tolerance kernel 5):')
    for day, sl in DAY_SLICES.items():
        for lbl, thr in zip(['G1+', 'G2+', 'G3+'], NOAA_G):
            yt = (a_s[:, sl] >= thr).astype(np.int8)
            s_r, _ = tss_pair((p_s[:, sl] >= TAU[lbl]).astype(np.int8), yt, kernel=KERNEL_D)
            s_f, _ = tss_pair((f_s[:, sl] >= thr).astype(np.int8), yt, kernel=KERNEL_D)
            s_o, _ = tss_pair((st_s[:, sl] >= thr).astype(np.int8), yt, kernel=KERNEL_D)
            print(f'  {day} {lbl}: rule {s_r:+.3f} | SWPC FIXED {s_f:+.3f} | stale {s_o:+.3f} (published)')
            oos_rows.append(dict(threshold=lbl, scope=day, rule_strict=s_r,
                                 swpc_fixed_strict=s_f, swpc_stale_strict=s_o))
    pd.DataFrame(oos_rows).to_csv(f'{OOS_ROOT}/oos_swpc_rescored_tables.csv', index=False)
    print(f'\nSaved: {ENS_DIR}/loocv_swpc_rescored.csv')
    print(f'Saved: {OOS_ROOT}/oos_predictions_swpcfix.csv')
    print(f'Saved: {OOS_ROOT}/oos_swpc_rescored_tables.csv', flush=True)


if __name__ == '__main__':
    main()
