"""Rescore Pipeline B with per-fold val-selected tau at ALL G-scales.

Preflight: experiments_log.md 2026-07-15 "tau-perfold".

The canonical LOOCV CSVs used per-fold val-selected tau_G1 but fixed
constants 30/46 at G2+/G3+. This harness re-derives tau_G2/tau_G3 per
fold on the same val split and rescores the pipeb columns. Regression
predictions only (LR ensemble columns are unaffected).

Gates:
  1. swept tau_G1 must equal the canonical CSV tau_g1 per fold
  2. fixed-tau (30/46) pipeb recomputation must match canonical CSV
     pipeb columns within 0.005 (proves harness alignment)

Output: runs/v14_agc_loocv_ensemble/loocv_pipeb_perfoldtau.csv
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
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong')
import paris_agc_loocv_fold_v14_ap_emu_pre2015 as P   # extends W.LOOCV_EVENTS to 32
import paris_agc_loocv_fold_v14_ap_emu as W
import v14_agc_loocv_ensemble as E
from dataset_leakfree import GatedDatasetLeakFree
from dataset_leakfree_ap_emu import ApEmulatorDataset
from dataset_leakfree_ap import AP_SCALE

ROOT = '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong'
ENS_DIR = f'{ROOT}/runs/v14_agc_loocv_ensemble'
CANON_ROOT = f'{ROOT}/runs/v14_agc_ap_emu_loocv'
PRE2015_ROOT = f'{ROOT}/runs/v14_agc_ap_emu_loocv_pre2015'
FIRST_NEW = P.FIRST_NEW_FOLD          # 26
NOAA_G = E.NOAA_G
N_BOOT = 1000

log = lambda *a: print('[tau-perfold]', *a, flush=True)


@torch.no_grad()
def predict(model, ds, idxs, bs=256):
    loader = DataLoader(Subset(ds, idxs), batch_size=bs, shuffle=False,
                        num_workers=4, pin_memory=True)
    preds, acts = [], []
    for batch in loader:
        x_img, x_anemo, x_dst, x_ap_hist, y_ap = [b.cuda() for b in batch[:5]]
        p = model(x_img, x_anemo, x_dst, x_ap_hist)
        preds.append((p * AP_SCALE).cpu().numpy())
        acts.append((y_ap * AP_SCALE).cpu().numpy())
    return np.concatenate(preds), np.concatenate(acts)


def main():
    torch.set_float32_matmul_precision('high')
    log('loading dataset...')
    base = GatedDatasetLeakFree(W.EMBEDDING_PATH, seq_len=W.SEQ_LEN_OVERRIDE,
                                forecast_horizon=W.HORIZON)
    ds = ApEmulatorDataset(base)
    event_to_idx = W.build_event_index(ds)
    canon = pd.concat([pd.read_csv(f'{ENS_DIR}/loocv_perscale_ensemble.csv'),
                       pd.read_csv(f'{ENS_DIR}/loocv_perscale_ensemble_pre2015.csv')],
                      ignore_index=True)

    rows, gate_fail = [], []
    for fold_id in range(len(W.LOOCV_EVENTS)):
        name = W.LOOCV_EVENTS[fold_id][0]
        safe = W.safe_event_name(name)
        root = PRE2015_ROOT if fold_id >= FIRST_NEW else CANON_ROOT
        ckpt = f'{root}/fold_{fold_id}_{safe}/baseline_run/best_model.ckpt'
        if not os.path.exists(ckpt):
            log(f'fold {fold_id}: missing ckpt, skip'); continue
        m = W.load_ckpt(ckpt)

        test_indices = event_to_idx[name]
        excl = W.lookback_overlap_indices(ds, name)
        train_pool = sorted(set(range(len(ds))) - excl)
        np.random.seed(E.SEED)
        pool_shuf = list(train_pool); np.random.shuffle(pool_shuf)
        split_at = int(0.85 * len(pool_shuf))
        val_idx = pool_shuf[split_at:]

        pred_va, act_va = predict(m, ds, sorted(val_idx))
        pred_te, act_te = predict(m, ds, test_indices)
        del m; torch.cuda.empty_cache()

        taus = {}
        for lbl, thr, fb in [('G1+', NOAA_G[0], 26), ('G2+', NOAA_G[1], 30),
                             ('G3+', NOAA_G[2], 46)]:
            taus[lbl] = E.sweep_tau_for(pred_va, act_va, thr, fallback=fb)

        c_fold = canon[canon.fold == fold_id]
        tau_g1_csv = int(c_fold.tau_g1.iloc[0])
        if taus['G1+'] != tau_g1_csv:
            gate_fail.append((fold_id, 'tau_g1', taus['G1+'], tau_g1_csv))

        storm = act_te.max(axis=1) >= NOAA_G[0]
        if storm.sum() == 0:
            log(f'fold {fold_id}: no storm windows, skip'); continue
        p_s, a_s = pred_te[storm], act_te[storm]
        fixed_tau = {'G1+': taus['G1+'], 'G2+': 30, 'G3+': 46}
        for lbl, thr in zip(['G1+', 'G2+', 'G3+'], NOAA_G):
            yt = (a_s >= thr).astype(np.int8)
            s_new, t_new = E.tss_pair((p_s >= taus[lbl]).astype(np.int8), yt)
            s_old, t_old = E.tss_pair((p_s >= fixed_tau[lbl]).astype(np.int8), yt)
            c = c_fold[c_fold.threshold == lbl]
            csv_pb = float(c.pipeb_strict.iloc[0]) if len(c) else np.nan
            if len(c) and abs(s_old - csv_pb) > 0.005:
                gate_fail.append((fold_id, lbl, s_old, csv_pb))
            rows.append(dict(fold=fold_id, event=name, threshold=lbl,
                             tau=taus[lbl], tau_fixed=fixed_tau[lbl],
                             pipeb_strict_perfold=s_new, pipeb_tol_perfold=t_new,
                             pipeb_strict_fixed=s_old, pipeb_tol_fixed=t_old,
                             csv_pipeb_strict=csv_pb,
                             n_storm=int(storm.sum()), pos_leads=int(yt.sum())))
        log(f'fold {fold_id:2d} {name}: taus G1={taus["G1+"]} G2={taus["G2+"]} G3={taus["G3+"]}')

    df = pd.DataFrame(rows)
    log('=== gates ===')
    if gate_fail:
        for f in gate_fail[:12]: log('  FAIL', f)
        raise SystemExit('gate FAILED — harness misaligned')
    log(f'  gates PASSED ({len(df)} cells; tau_g1 + fixed-tau reproduction)')
    df.to_csv(f'{ENS_DIR}/loocv_pipeb_perfoldtau.csv', index=False)

    resc = pd.read_csv(f'{ENS_DIR}/loocv_swpc_rescored.csv')
    rng = np.random.default_rng(42)
    def med_ci(v):
        v = pd.Series(v).dropna().values
        b = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(N_BOOT)]
        return np.median(v), np.percentile(b, 2.5), np.percentile(b, 97.5)

    print('\n' + '=' * 96)
    print('32-FOLD LOOCV — Pipeline B with per-fold tau at all scales vs fixed 30/46 (SWPC = corrected)')
    print('=' * 96)
    for lbl in ['G1+', 'G2+', 'G3+']:
        sub = df[df.threshold == lbl]
        ssub = resc[resc.threshold == lbl]
        for col, tag in [('pipeb_strict_perfold', 'per-fold tau strict'),
                         ('pipeb_strict_fixed', 'fixed tau strict   '),
                         ('pipeb_tol_perfold', 'per-fold tau tol   '),
                         ('pipeb_tol_fixed', 'fixed tau tol      ')]:
            m, lo, hi = med_ci(sub[col])
            print(f'  {lbl} {tag} {m:+.3f} [{lo:+.3f},{hi:+.3f}]')
        m, lo, hi = med_ci(ssub.swpc_strict_fixed)
        print(f'  {lbl} SWPC corrected strict {m:+.3f} [{lo:+.3f},{hi:+.3f}]')
        taus = sorted(sub.tau.unique())
        print(f'  {lbl} tau range across folds: {taus}')
    print(f'\nSaved: {ENS_DIR}/loocv_pipeb_perfoldtau.csv', flush=True)


if __name__ == '__main__':
    main()
