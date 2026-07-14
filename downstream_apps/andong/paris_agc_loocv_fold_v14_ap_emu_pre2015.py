"""Pre-2015 catalog extension — 6 new LOOCV folds (Events #28-#33).

Wraps paris_agc_loocv_fold_v14_ap_emu.py unchanged: same H5, architecture,
loss, sampler, seed, epochs.  Only the event catalog is extended (fold ids
26..31) and the output root is separate so the canonical 26-fold results
in runs/v14_agc_ap_emu_loocv are never touched.

Data audit (2026-07-14): all six windows pass the kept-event coverage bar
(AIA >=89%, C3 >=97%, GONG H-alpha at the pre-2023 diurnal ~30-47% level,
SWPC leak-free archive complete, OMNI ap / physics / Dst 100%).
Excluded for cause: 2011-08-05 + 2011-10-24 (predate SWPC archive),
2014-02-19 (267 h GONG outage), 2012-03-07 + 2015-03-17 (marginal; not run).

Usage:
  python paris_agc_loocv_fold_v14_ap_emu_pre2015.py --dryrun          # audit only
  python paris_agc_loocv_fold_v14_ap_emu_pre2015.py --fold 26 --gpu 0 # train one fold
"""
from __future__ import annotations
import os, sys, argparse

sys.path.insert(0, '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong')
import paris_agc_loocv_fold_v14_ap_emu as W

# Catalog-style windows (buildup -> recovery), same convention as Events #2-#27.
EXTRA_EVENTS = [
    ("Event #28", "2012-04-18 18:00:00", "2012-05-04 21:00:00"),  # 2012-04-24, peak ap 111
    ("Event #29", "2012-07-10 00:00:00", "2012-07-25 12:00:00"),  # 2012-07-15, peak ap 132
    ("Event #30", "2012-10-03 06:00:00", "2012-10-18 21:00:00"),  # 2012-10-09, peak ap 111
    ("Event #31", "2013-03-12 06:00:00", "2013-03-26 21:00:00"),  # St. Patrick 2013, peak ap 111
    ("Event #32", "2013-05-27 00:00:00", "2013-06-10 12:00:00"),  # 2013-06-01, peak ap 132
    ("Event #33", "2013-09-27 00:00:00", "2013-10-11 21:00:00"),  # 2013-10-02, peak ap 179
]
EXPECTED_PEAK_AP = {"Event #28": 111, "Event #29": 132, "Event #30": 111,
                    "Event #31": 111, "Event #32": 132, "Event #33": 179}

W.LOOCV_EVENTS = list(W.LOOCV_EVENTS) + EXTRA_EVENTS
W.OUT_ROOT = os.environ.get(
    'V14_OUT_ROOT',
    '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong/runs/v14_agc_ap_emu_loocv_pre2015')
FIRST_NEW_FOLD = len(W.LOOCV_EVENTS) - len(EXTRA_EVENTS)   # 26


def dryrun():
    """Build the dataset once; audit every new fold without training."""
    import numpy as np
    import pandas as pd
    from dataset_leakfree import GatedDatasetLeakFree
    from dataset_leakfree_ap_emu import ApEmulatorDataset
    from dataset_leakfree_ap import AP_STORM_THRESHOLD

    base = GatedDatasetLeakFree(W.EMBEDDING_PATH, seq_len=W.SEQ_LEN_OVERRIDE,
                                forecast_horizon=W.HORIZON)
    full_ap = ApEmulatorDataset(base)
    event_to_idx = W.build_event_index(full_ap)
    swpc = base.swpc_forecasts.numpy()
    ok = True
    for name, start, end in EXTRA_EVENTS:
        idxs = event_to_idx[name]
        n = len(idxs)
        peak = float(full_ap.ap_grid_raw[idxs, :].max()) if n else float('nan')
        excl = W.lookback_overlap_indices(full_ap, name)
        # storm-window issues (>=1 lead with ap >= G1 entry 39)
        grid = full_ap.ap_grid_raw[idxs, :].numpy() if n else np.zeros((0, W.HORIZON))
        n_storm = int((grid.max(axis=1) >= 39).sum())
        # leak-free SWPC coverage on the held-out issues (share with any nonzero lead)
        cov = float((swpc[idxs] > 0).any(axis=1).mean()) if n else 0.0
        _, diag = W.pdf_sampler_weights(full_ap, sorted(set(range(len(full_ap))) - excl))
        exp = EXPECTED_PEAK_AP[name]
        line_ok = (n > 0 and abs(peak - exp) < 1e-6 and n_storm > 0 and cov >= 0.8
                   and 0.5 <= diag['storm_share_train'] * 100 <= 4.0)
        ok &= line_ok
        print(f'{name}  [{start} .. {end}]  n_test={n}  n_storm_issues={n_storm}  '
              f'peak_ap={peak:.0f} (expect {exp})  excl={len(excl)}  '
              f'swpc_cov={cov*100:.1f}%  storm_share_train={diag["storm_share_train"]*100:.2f}%  '
              f'{"OK" if line_ok else "FAIL"}', flush=True)
    print('DRYRUN', 'PASS' if ok else 'FAIL', flush=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=None)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dryrun', action='store_true')
    args = ap.parse_args()
    if args.dryrun:
        sys.exit(dryrun())
    assert args.fold is not None and args.fold >= FIRST_NEW_FOLD, \
        f'this wrapper only runs new folds {FIRST_NEW_FOLD}..{len(W.LOOCV_EVENTS)-1}'
    W.run_one_fold(args.fold, args.gpu, seed=args.seed)


if __name__ == '__main__':
    main()
