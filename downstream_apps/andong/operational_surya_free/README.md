# Operational Surya-Free 48h Dst Forecaster (0421)

A **fully deterministic**, single-checkpoint 48-hour-ahead Dst forecasting model.
No SDO/Surya imagery required at inference time.

## Files

| File | Purpose |
|------|---------|
| `surya_free_forecaster.py` | **Option A** — ML-only forecaster (`SuryaFreeForecaster`) |
| `surya_free_hybrid.py` | **Option B** — SET+ML hybrid AND-gate (`SuryaFreeHybridForecaster`) |
| `surya_free_tnr_priority.py` | **Option C** — TNR-priority FC-fine-tuned (`SuryaFreeTNRPriorityForecaster`) |
| `fold_17_best_model-v1.ckpt` | Base trained model (used by Options A & B) |
| `fold_17_fc_tnr.ckpt` | TNR-priority fine-tuned FC head (Option C) |
| `mean_surya.npy` | Fixed mean Surya embedding, shape (32, 1280) |
| `validate.py` | Reproduce Option A metrics |
| `validate_hybrid.py` | Reproduce Option B metrics |
| `validate_tnr_priority.py` | Reproduce Option C metrics |
| `generate_report.py` | Build the PDF report |
| `Operational_Surya_Free_Report.pdf` | Full report |
| `README.md` | This file |

## Three deployment options

| Option | Module | Description | Beats SET on |
|--------|--------|-------------|--------------|
| **A. ML-only** | `SuryaFreeForecaster` | Trained model + fixed mean Surya | TPR only |
| **B. Hybrid AND-gate** | `SuryaFreeHybridForecaster` | SET default, ML overrides only on confident deep-storm prediction during strong flares | TPR only (matches SET TNR) |
| **C. TNR-priority FT** | `SuryaFreeTNRPriorityForecaster` | Same backbone as A; FC head re-fine-tuned with extra non-SET quiet samples and a false-alarm penalty | **Both TPR and TNR** |

## Why deterministic?

The Surya tower is fed a **fixed mean vector** every time — no random draws,
no rng state, no batch-order dependence. Identical inputs always produce the
same output.

## Quick start

### Option A — ML-only forecaster

```python
from surya_free_forecaster import SuryaFreeForecaster
import numpy as np

fc = SuryaFreeForecaster()

# Inputs for one forecast (all REAL — no Surya data needed):
x_lasco = ...   # (32, 1280)  SOHO/LASCO C2 embeddings (3h cadence, oldest first)
x_anemo = ...   # (32, 8)     physics features
x_dst   = ...   # (32, 1)     past Dst scaled nT/100

dst_48h = fc.predict(x_lasco, x_anemo, x_dst)
print(f"Dst at T+48h: {dst_48h:.1f} nT")
```

### Option B — Hybrid AND-gate (SET + ML)

```python
from surya_free_hybrid import SuryaFreeHybridForecaster

fc = SuryaFreeHybridForecaster()

# Same inputs as Option A, plus SET's dst_2d forecast at T
result = fc.predict(x_lasco, x_anemo, x_dst, set_dst_2d=set_forecast_nT)
# result: {'dst_48h': float, 'source': 'ML' or 'SET', 'ml_pred': float, 'gate_active': bool}

print(f"Dst at T+48h: {result['dst_48h']:.1f} nT  (source={result['source']})")
```

The hybrid uses SET's prediction by default and only overrides with the ML
forecast when all three conditions hold:
1. Max flare class in past 96 h ≥ 3.7 (strong C / M / X class)
2. ML prediction ≤ −180 nT (confident deep storm)
3. SET prediction ≥ −20 nT (SET wasn't already flagging the storm)

### Option C — TNR-priority FC fine-tuned forecaster

```python
from surya_free_tnr_priority import SuryaFreeTNRPriorityForecaster

fc = SuryaFreeTNRPriorityForecaster()

# Same inputs as Option A — drop-in replacement, no SET input needed:
dst_48h = fc.predict(x_lasco, x_anemo, x_dst)
print(f"Dst at T+48h: {dst_48h:.1f} nT")
```

Same API and same backbone as Option A, but loads the FC-fine-tuned
checkpoint (`fold_17_fc_tnr.ckpt`). Beats SET on both TPR and TNR across
the full N=6,692 set.

Physics features (8 per timestep):

1. Flare class (A=1, B=2, C=3, M=4, X=5, + fractional)
2. Heliographic latitude (deg)
3. Heliographic longitude (deg)
4. X-ray flux
5. Flare duration (hours above half-flux)
6. Current Dst (nT)
7. Distance from disk center (deg)
8. Flare energy (class × duration)

## Validation

Apple-to-apple comparison with SET. Ground truth: `Dst_per3` (3h backward
rolling average of hourly OMNI Dst, the model's training label) at T+48h.

### View A: Event-window samples (N=2,220) — storm-period performance

| Model | TPR | TNR | TP | FP | TSS |
|-------|-----|-----|----|----|----|
| SET (Anemomilos) | 0.167 | 0.938 | 19 | 131 | +0.104 |
| **Deployed model** | **0.316** | **0.962** | **36** | **80** | **+0.278** |

Beats SET on both TPR (+89%) and TNR (39% fewer false alarms).

### View B: Truly unseen quiet time (N=4,472) — outside all event windows

| Model | FP | TN | TNR | FPR |
|-------|----|----|-----|-----|
| SET | 34 | 4,438 | **0.992** | **0.76%** |
| Deployed model | 213 | 4,259 | 0.952 | 4.76% |

SET's physics-driven sparse activation gives it stronger false-alarm
suppression on long calm periods. This is where Option B (the hybrid
gate) addresses the gap.

### View C: Combined (N=6,692) — realistic operational scenario

| Model | TPR | TNR | TP | FP | TSS |
|-------|-----|-----|----|----|----|
| SET | 0.167 | 0.975 | 19 | 165 | +0.142 |
| **Option A: ML-only** | **0.316** | 0.955 | 36 | 293 | +0.271 |
| **Option B: Hybrid AND-gate** | 0.193 | 0.975 | 22 | 165 | +0.168 |
| **Option C: TNR-priority FT** | **0.281** | **0.978** | **32** | **146** | **+0.259** |

- Option A has the highest TPR (2× SET) but extra FPs in unseen quiet.
- Option B **matches SET's TNR exactly** (FP tied at 165) while adding 3 TPs —
  the safer operational choice when false-alarm budget is strict. Only 0.04%
  of samples trigger the ML override.
- **Option C strictly beats SET on both TPR and TNR** — +13 TPs (69% more
  storms caught) and −19 FPs (11% fewer false alarms). This is the first
  single-checkpoint deterministic model to dominate SET on both axes across
  the full operational set.

## Reproduce the metrics

```bash
python validate.py                 # Option A: ML-only
python validate_hybrid.py          # Option B: Hybrid AND-gate
python validate_tnr_priority.py    # Option C: TNR-priority FT
```

## Real-time deployment data sources

| Channel | Source | Latency |
|---------|--------|---------|
| LASCO C2 embeddings | SOHO/LASCO + the embedding network | ~hours |
| Physics features | GOES X-ray flux + SWPC flare catalog | near real-time |
| Past Dst | Kyoto OMNI Dst (3h rolling avg) | ~few hours |
| Surya slot | `mean_surya.npy` (fixed) | instant |

## Caveats

1. **Quiet-time false-alarm rate (Option A)**: View B shows Option A triggers
   more often than SET during long calm periods. Option B (hybrid) or Option C
   (TNR-priority FT) are recommended when false-alarm budget is strict.
2. **48h lead time only**: shorter lead times have not been validated.
3. **Training data ends November 2024**. Monitor performance as time advances
   beyond the training window.
4. **Option C training data**: the FC fine-tune uses extra non-SET quiet
   samples for training — SET-matched samples are held out for validation, so
   the Option C vs SET comparison remains fair.
