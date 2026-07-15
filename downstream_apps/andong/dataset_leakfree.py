"""Leak-free SWPC dataset.

Same as GatedDataset but rebuilds the swpc_forecasts tensor using strict
pre-issue keying:
  - SWPC publishes at ~12:30 UTC.
  - For V12 issue_time t with hour <= 12: use forecast issued day-(D-1).
  - For V12 issue_time t with hour >= 15: use forecast issued day-D.
  - For target tt = t + lead, look up SWPC value in (chosen forecast)'s
    day1/day2/.../day7 bins.

Output: same tuple shape as GatedDataset:
    (x_img, x_anemo, x_dst, y_residual, y_swpc, is_storm)
But the y_swpc / y_residual are now operationally-honest (no future SWPC info).
"""
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '/media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong')
from Main_kp_v11_3d_strat import (
    SuryaSpaceWeatherDatasetV11, KP_SCALE_FACTOR, SWPC_PATH, HORIZON, SEQ_LEN,
)
from Main_kp_v12_gated import GatedDataset
from Main_kp_v15_recurrence import RECURRENCE_LAG_STEPS, RECURRENCE_WINDOW


class GatedDatasetLeakFree(GatedDataset):
    """GatedDataset with leak-free SWPC tensor rebuilt at construction."""

    def __init__(self, h5_filepath, seq_len=SEQ_LEN, forecast_horizon=HORIZON,
                 publication_cutoff_hour=12):
        # Build the original (leaked) dataset first via parent
        super().__init__(h5_filepath, seq_len=seq_len, forecast_horizon=forecast_horizon)
        # Now rebuild swpc_forecasts with leak-free keying, replacing the leaked tensor
        self._rebuild_swpc_leakfree(publication_cutoff_hour=publication_cutoff_hour)

    def _rebuild_swpc_leakfree(self, publication_cutoff_hour=12):
        print(f"\n[leak-free] Rebuilding SWPC tensor with strict pre-issue keying "
              f"(cutoff hour={publication_cutoff_hour})...", flush=True)

        swpc_raw = pd.read_csv(SWPC_PATH)
        swpc_raw['issue_date'] = pd.to_datetime(swpc_raw['Forecast Day'], format='%m/%d/%Y')
        kp_cols  = ['00-03 UTC','03-06 UTC','06-09 UTC','09-12 UTC',
                    '12-15 UTC','15-18 UTC','18-21 UTC','21-00 UTC']
        kp_hours = [0, 3, 6, 9, 12, 15, 18, 21]

        swpc_lookup = {}
        for _, row in swpc_raw.iterrows():
            d = int(row['Day after forecast'].replace('day', ''))
            target_start = row['issue_date'] + pd.Timedelta(days=d - 1)
            for col, h in zip(kp_cols, kp_hours):
                v = row[col]
                if pd.isna(v): continue
                t = target_start + pd.Timedelta(hours=h)
                swpc_lookup.setdefault(row['issue_date'], {})[t] = float(v)

        N = len(self.df)
        new_swpc = np.zeros((N, self.forecast_horizon), dtype=np.float32)
        new_valid = np.zeros((N, self.forecast_horizon), dtype=bool)
        n_matched = 0
        n_dminus1 = 0   # rows that used D-1 forecast
        n_dzero   = 0   # rows that used D forecast
        n_no_lookup = 0
        for i in range(N):
            t = pd.Timestamp(self.timestamps[i])
            if t.hour <= publication_cutoff_hour:
                key = t.normalize() - pd.Timedelta(days=1)
                used_dminus1 = True
            else:
                key = t.normalize()
                used_dminus1 = False
            swpc_for_day = swpc_lookup.get(key)
            if swpc_for_day is None:
                n_no_lookup += 1
                continue
            for step in range(self.forecast_horizon):
                tt    = t + pd.Timedelta(hours=(step + 1) * 3)
                tt_3h = tt.floor('3h')
                if tt_3h in swpc_for_day:
                    new_swpc[i, step] = swpc_for_day[tt_3h] * 10.0
                    new_valid[i, step] = True
                    n_matched += 1
            if used_dminus1: n_dminus1 += 1
            else:            n_dzero += 1

        coverage_overall = n_matched / (N * self.forecast_horizon) * 100
        coverage_step24  = (new_swpc[:, -1] > 0).sum() / N * 100
        print(f"  [leak-free] coverage overall: {coverage_overall:.1f}%  "
              f"step-24 coverage: {coverage_step24:.1f}%", flush=True)
        print(f"  [leak-free] used D-1 (early issues): {n_dminus1}    "
              f"used D (late issues): {n_dzero}    "
              f"no lookup: {n_no_lookup}", flush=True)

        self.swpc_forecasts = torch.tensor(new_swpc, dtype=torch.float32) / KP_SCALE_FACTOR
        # True where a bulletin bin actually matched; zeros in swpc_forecasts
        # at ~valid positions are coverage gaps, not quiet forecasts.
        self.swpc_valid = torch.from_numpy(new_valid)
        print(f"[leak-free] SWPC tensor replaced.\n", flush=True)


class RecurrenceDatasetLeakFree(GatedDatasetLeakFree):
    """GatedDatasetLeakFree + per-timestep recurrence features (Kp at t-27d to t-28d).

    Mirrors Main_kp_v15_recurrence.RecurrenceDataset but inherits the leak-free
    SWPC tensor from GatedDatasetLeakFree.
    """
    def __init__(self, h5_filepath, seq_len=SEQ_LEN, forecast_horizon=HORIZON,
                 publication_cutoff_hour=12):
        super().__init__(h5_filepath, seq_len=seq_len,
                          forecast_horizon=forecast_horizon,
                          publication_cutoff_hour=publication_cutoff_hour)
        print(f"[leak-free+rec] Building recurrence features "
              f"(lag={RECURRENCE_LAG_STEPS} steps, window={RECURRENCE_WINDOW})...", flush=True)
        kp_series = self.continuous_dst.squeeze(-1).numpy()
        N = kp_series.shape[0]
        recurrence = np.zeros((N, RECURRENCE_WINDOW), dtype=np.float32)
        for i in range(N):
            for k in range(RECURRENCE_WINDOW):
                src = i - RECURRENCE_LAG_STEPS - k
                if 0 <= src < N:
                    recurrence[i, k] = kp_series[src]
        n_valid = (recurrence != 0).any(axis=1).sum()
        print(f"  recurrence: shape={recurrence.shape}, valid samples: {n_valid}/{N}", flush=True)
        self.recurrence = torch.from_numpy(recurrence)

    def __getitem__(self, idx):
        x_img, x_anemo, x_dst, y_residual, y_swpc, is_storm = super().__getitem__(idx)
        rec = self.recurrence[idx : idx + self.seq_len]
        return x_img, x_anemo, x_dst, rec, y_residual, y_swpc, is_storm
