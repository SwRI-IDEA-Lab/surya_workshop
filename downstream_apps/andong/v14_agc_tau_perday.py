"""Per-day tau selection for the LIVE-Ap threshold rule (Pipeline B).

Preflight: experiments_log.md 2026-07-15 11:57 UTC.

Selects tau(day, scale) on the cutoff-run validation split (never the OOS
window), then scores the OOS forward test per day against the global-tau
rule and leak-free SWPC-Ap. Alignment checks reproduce the canonical
global tau (26/30/46) and the manuscript per-day table before the new
numbers count.

Output: runs/v14_agc_cutoff20240831/tau_perday/tau_perday_oos.csv
"""
from __future__ import annotations
import os, sys, argparse

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
from scipy.ndimage import maximum_filter1d
sys.path.insert(0, '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong')
from Main_kp_v11_3d_strat import HORIZON, EMBEDDING_PATH
from dataset_leakfree import GatedDatasetLeakFree
from dataset_leakfree_ap_emu import ApEmulatorDataset
from paris_agc_loocv_fold_v14_xgb_hybrid import HStateExtractorReg
from paris_agc_loocv_fold_v14_xgb_alpha_sweep import extract_features

RUN_ROOT = ('/media/faraday/andong/Workspace/surya_workshop/downstream_apps/'
            'andong/runs/v14_agc_cutoff20240831')
CKPT = f'{RUN_ROOT}/train/best_model.ckpt'
SPLIT = f'{RUN_ROOT}/split.npz'
OUT_DIR = f'{RUN_ROOT}/tau_perday'
os.makedirs(OUT_DIR, exist_ok=True)

G_BOUNDS = [('G1+', 39), ('G2+', 67), ('G3+', 111)]
GLOBAL_TAU_EXPECTED = {'G1+': 26, 'G2+': 30, 'G3+': 46}
MANUSCRIPT_PERDAY = {  # tab:v14agc_perday, global-tau rule, strict
    ('G1+', 'D1'): 0.249, ('G1+', 'D2'): 0.041, ('G1+', 'D3'): 0.074,
    ('G2+', 'D1'): 0.252, ('G2+', 'D2'): 0.019, ('G2+', 'D3'): 0.088,
    ('G3+', 'D1'): 0.251, ('G3+', 'D2'): -0.006, ('G3+', 'D3'): 0.000,
}
DAY_SLICES = {'D1': slice(0, 8), 'D2': slice(8, 16), 'D3': slice(16, 24)}
KERNEL_D = 5                 # within-day +/-6h tolerance (existing convention)
TAU_SWEEP = list(range(2, 200, 2))
MIN_POS = 10
N_BOOT = 1000

log = lambda *a: print('[tau-perday]', *a, flush=True)


def tss(yp, yt):
    yp = yp.reshape(-1).astype(np.int8); yt = yt.reshape(-1).astype(np.int8)
    tp = int(((yp == 1) & (yt == 1)).sum()); fn = int(((yp == 0) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum()); tn = int(((yp == 0) & (yt == 0)).sum())
    return (tp / max(1, tp + fn)) - (fp / max(1, fp + tn))


def tpr_fpr(yp, yt):
    yp = yp.reshape(-1); yt = yt.reshape(-1)
    tp = ((yp == 1) & (yt == 1)).sum(); fn = ((yp == 0) & (yt == 1)).sum()
    fp = ((yp == 1) & (yt == 0)).sum(); tn = ((yp == 0) & (yt == 0)).sum()
    return tp / max(1, tp + fn), fp / max(1, fp + tn)


def tol_day(y_2d):
    return maximum_filter1d(y_2d, size=KERNEL_D, axis=1, mode='nearest')


def sweep_tau(p_2d, yt_2d):
    """argmax-strict-TSS tau over the sweep grid."""
    best = (-999, None)
    for tau in TAU_SWEEP:
        s = tss((p_2d >= tau).astype(np.int8), yt_2d)
        if s > best[0]:
            best = (s, tau)
    return best[1], best[0]


def main():
    device = 'cuda:0'
    torch.set_float32_matmul_precision('high')

    # ---- val predictions (for tau selection) -----------------------------
    log('Loading dataset...')
    base = GatedDatasetLeakFree(EMBEDDING_PATH, seq_len=56, forecast_horizon=HORIZON)
    ds = ApEmulatorDataset(base)
    sp = np.load(SPLIT)
    val_indices = sp['val_indices'].tolist()
    log(f'val n={len(val_indices):,}')

    log('Loading ckpt + extracting val predictions...')
    m = HStateExtractorReg(img_dim=3840, anemo_dim=8, dst_dim=1,
                           hidden_dim=128, num_layers=2, dropout=0.3,
                           forecast_horizon=HORIZON).to(device)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    sd = {k.replace('model.', '', 1): v for k, v in ck['state_dict'].items()
          if k.startswith('model.')}
    m.load_state_dict(sd, strict=True); m.eval()
    _, _, mva = extract_features(m, ds, val_indices, device)
    dfv = pd.DataFrame(mva).sort_values(['issue_time', 'lead_h']).reset_index(drop=True)
    nv = len(dfv) // HORIZON
    pred_v = dfv['pred_ap_v14'].values.reshape(nv, HORIZON)
    act_v = dfv['actual_ap'].values.reshape(nv, HORIZON)
    storm_v = act_v.max(axis=1) >= 39
    p_vs, a_vs = pred_v[storm_v], act_v[storm_v]
    log(f'val issues={nv}  storm issues={int(storm_v.sum())}')

    # ---- check 1: reproduce global tau -----------------------------------
    log('=== check 1: global tau on val (expect 26/30/46) ===')
    global_tau = {}
    for lbl, g in G_BOUNDS:
        yt = (a_vs >= g).astype(np.int8)
        t, s = sweep_tau(p_vs, yt)
        global_tau[lbl] = t
        log(f'  {lbl}: tau={t}  val strict={s:+.4f}  n_pos={int(yt.sum())}')
    assert global_tau == GLOBAL_TAU_EXPECTED, \
        f'global tau reproduction FAILED: {global_tau} != {GLOBAL_TAU_EXPECTED}'
    log('  check 1 PASSED')

    # ---- per-day tau selection on val -------------------------------------
    log('=== per-day tau selection on val ===')
    perday_tau = {}
    for day, sl in DAY_SLICES.items():
        for lbl, g in G_BOUNDS:
            yt = (a_vs[:, sl] >= g).astype(np.int8)
            n_pos = int(yt.sum())
            if n_pos < MIN_POS:
                perday_tau[(day, lbl)] = (global_tau[lbl], n_pos, 'fallback')
                log(f'  {day} {lbl}: n_pos={n_pos} < {MIN_POS} -> FALLBACK to global tau={global_tau[lbl]}')
                continue
            t, s = sweep_tau(p_vs[:, sl], yt)
            flag = 'boundary' if t in (TAU_SWEEP[0], TAU_SWEEP[-1]) else 'ok'
            perday_tau[(day, lbl)] = (t, n_pos, flag)
            log(f'  {day} {lbl}: tau={t}  val strict={s:+.4f}  n_pos={n_pos}  [{flag}]')

    # ---- OOS scoring -------------------------------------------------------
    log('Loading OOS predictions...')
    df = pd.read_csv(f'{RUN_ROOT}/oos_predictions.csv')
    df = df.sort_values(['issue_time', 'lead_h']).reset_index(drop=True)
    nte = len(df) // HORIZON
    pred_t = df['pred_ap_v14'].values.reshape(nte, HORIZON)
    act_t = df['actual_ap'].values.reshape(nte, HORIZON)
    swpc_t = df['swpc_ap'].values.reshape(nte, HORIZON)
    storm_t = act_t.max(axis=1) >= 39
    p_ts, a_ts, sw_ts = pred_t[storm_t], act_t[storm_t], swpc_t[storm_t]
    log(f'OOS issues={nte}  storm issues={int(storm_t.sum())} (expect 629)')

    rng = np.random.default_rng(42)
    rows = []
    check2_fail = []
    for day, sl in DAY_SLICES.items():
        for lbl, g in G_BOUNDS:
            yt = (a_ts[:, sl] >= g).astype(np.int8)
            yp_glob = (p_ts[:, sl] >= global_tau[lbl]).astype(np.int8)
            tau_d = perday_tau[(day, lbl)][0]
            yp_day = (p_ts[:, sl] >= tau_d).astype(np.int8)
            yp_swpc = (sw_ts[:, sl] >= g).astype(np.int8)

            s_glob = tss(yp_glob, yt); s_day = tss(yp_day, yt); s_swpc = tss(yp_swpc, yt)
            t_glob = tss(tol_day(yp_glob), tol_day(yt))
            t_day = tss(tol_day(yp_day), tol_day(yt))
            t_swpc = tss(tol_day(yp_swpc), tol_day(yt))
            tpr_d, fpr_d = tpr_fpr(yp_day, yt)
            tpr_g, fpr_g = tpr_fpr(yp_glob, yt)

            # check 2: global-tau per-day strict must match manuscript table
            ref = MANUSCRIPT_PERDAY[(lbl, day)]
            if abs(s_glob - ref) > 0.005:
                check2_fail.append((lbl, day, s_glob, ref))

            # issue-bootstrap CI of (per-day - global) strict delta
            n = len(yt)
            deltas = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.integers(0, n, n)
                deltas[b] = tss(yp_day[idx], yt[idx]) - tss(yp_glob[idx], yt[idx])
            lo, hi = np.percentile(deltas, [2.5, 97.5])

            rows.append(dict(day=day, threshold=lbl,
                             tau_global=global_tau[lbl], tau_day=tau_d,
                             tau_day_flag=perday_tau[(day, lbl)][2],
                             val_n_pos=perday_tau[(day, lbl)][1],
                             swpc_strict=s_swpc, glob_strict=s_glob, day_strict=s_day,
                             delta_strict=s_day - s_glob,
                             delta_lo=lo, delta_hi=hi,
                             swpc_tol=t_swpc, glob_tol=t_glob, day_tol=t_day,
                             tpr_day=tpr_d, fpr_day=fpr_d,
                             tpr_glob=tpr_g, fpr_glob=fpr_g,
                             pos_leads=int(yt.sum())))

    log('=== check 2: manuscript per-day reproduction under global tau ===')
    if check2_fail:
        for lbl, day, got, ref in check2_fail:
            log(f'  FAIL {lbl} {day}: computed {got:+.4f} vs manuscript {ref:+.3f}')
        raise SystemExit('check 2 FAILED — alignment bug, numbers void')
    log('  check 2 PASSED (all 9 cells within 0.005 of tab:v14agc_perday)')

    out = pd.DataFrame(rows)
    out.to_csv(f'{OUT_DIR}/tau_perday_oos.csv', index=False)

    print('\n' + '=' * 100)
    print('PER-DAY tau (val-selected) vs global tau vs SWPC-Ap — OOS storm-window strict TSS')
    print('=' * 100)
    for day in DAY_SLICES:
        print(f'\n--- {day} ---')
        sub = out[out.day == day]
        print(sub[['threshold', 'tau_global', 'tau_day', 'tau_day_flag',
                   'swpc_strict', 'glob_strict', 'day_strict', 'delta_strict',
                   'delta_lo', 'delta_hi', 'tpr_day', 'fpr_day', 'pos_leads']]
              .to_string(index=False, float_format='%.4f'))
    print(f'\nSaved: {OUT_DIR}/tau_perday_oos.csv', flush=True)


if __name__ == '__main__':
    main()
