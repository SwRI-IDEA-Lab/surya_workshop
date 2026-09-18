"""Balanced active / quiet sampling for the SEP-from-PSP task.

A sample is labelled by the SEP intensity at its PSP hour, with two thresholds and a gap
between them:

    intensity >  active_above   ->  ACTIVE   (is_event = 1)
    intensity <  quiet_below    ->  QUIET    (is_event = 0)
    anything in between         ->  DROPPED, never sampled

The middle band is excluded on purpose: those hours are neither clearly an SEP enhancement
nor clearly background, and including them on either side would blur exactly the contrast
the task is about. Dropping them costs nothing -- the quiet class has candidates to spare.

Two properties the rest of the app depends on:

- **Draws are nested.** Each pool is shuffled once from the seeded RNG, in an order that
  does not depend on ``n_samples``, and the draw is a prefix. So at one seed the 50-sample
  training set is a strict subset of the 100-sample one, and a 50/100/200/500 ladder is one
  growing dataset rather than four unrelated draws. Anything that makes a pool size or an
  RNG draw depend on ``n_samples`` breaks this.
- **One sample per Surya frame.** Several active regions share a PSP hour, and their rows
  can match the same SDO frame; only the closest match is kept, so a frame is never in the
  set twice.

Everything here is a pure function of DataFrames, so it can be tested without Surya data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

PSP_TIME = "time"

# Catalog-side direction for pd.merge_asof. ds_match_direction is phrased from the Surya
# side ("forward": the catalog time is at or after the Surya frame, i.e. causal), while
# here the catalog is the left table, so each direction flips.
_CATALOG_DIRECTION = {"forward": "backward", "backward": "forward", "nearest": "nearest"}


def match_catalog_to_surya(
    catalog: pd.DataFrame,
    surya_times: list,
    time_column: str,
    tolerance: str | None,
    direction: str = "forward",
) -> pd.DataFrame:
    """Attach the Surya frame for each catalog row; drop rows with no frame in tolerance.

    Adds ``valid_indices`` (the Surya timestep), ``ds_index`` (the catalog time used for
    matching) and ``index_delta`` (their absolute gap).
    """
    catalog = catalog.copy()
    catalog["ds_index"] = pd.to_datetime(catalog[time_column], format="ISO8601").astype(
        "datetime64[ns]"
    )
    catalog[PSP_TIME] = pd.to_datetime(catalog[PSP_TIME], format="ISO8601").astype(
        "datetime64[ns]"
    )
    surya = pd.DataFrame(
        {"valid_indices": pd.to_datetime(pd.Series(surya_times)).astype("datetime64[ns]")}
    ).sort_values("valid_indices")

    matched = pd.merge_asof(
        catalog.sort_values("ds_index"),
        surya,
        left_on="ds_index",
        right_on="valid_indices",
        direction=_CATALOG_DIRECTION[direction],
        tolerance=pd.Timedelta(tolerance) if tolerance is not None else None,
    )
    matched = matched.dropna(subset=["valid_indices"])
    matched["index_delta"] = (matched["valid_indices"] - matched["ds_index"]).abs()
    return matched.reset_index(drop=True)


def _one_row_per_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per Surya frame, keeping the closest catalog match."""
    return rows.sort_values("index_delta", kind="stable").drop_duplicates("valid_indices")


def active_candidates(
    matched: pd.DataFrame, label_column: str, active_above: float
) -> pd.DataFrame:
    """Rows whose intensity is above ``active_above``, one per Surya frame."""
    return _one_row_per_frame(matched[matched[label_column] > active_above]).reset_index(drop=True)


def quiet_candidates(
    matched: pd.DataFrame,
    label_column: str,
    quiet_below: float,
    active_above: float,
    buffer: str = "1d",
) -> pd.DataFrame:
    """Rows below ``quiet_below`` and at least ``buffer`` in PSP time from any active hour.

    The buffer is what keeps the two classes apart in time as well as in value: the hours
    on the flank of an SEP enhancement are below the quiet threshold while the Sun is still
    in the state that produced it, so an SDO frame taken there looks active while its label
    says quiet. ``buffer`` is measured against every active hour in ``matched``, not only
    the ones that end up sampled.
    """
    quiet = matched[matched[label_column] < quiet_below]
    active_times = matched.loc[matched[label_column] > active_above, PSP_TIME].to_numpy()
    if len(active_times) and len(quiet):
        pad = pd.Timedelta(buffer).to_timedelta64()
        gap = np.abs(quiet[PSP_TIME].to_numpy()[:, None] - active_times[None, :]).min(axis=1)
        quiet = quiet.loc[gap >= pad]
    return _one_row_per_frame(quiet).reset_index(drop=True)


def sample_balanced(
    matched: pd.DataFrame,
    n_samples: int | None,
    seed: int,
    label_column: str,
    active_above: float,
    quiet_below: float,
    non_event_buffer: str = "1d",
) -> pd.DataFrame:
    """Draw a half-active / half-quiet sample, one row per Surya frame.

    ``n_samples`` gives N//2 active and N - N//2 quiet rows (an odd extra is quiet).
    ``n_samples=None`` uses every active candidate plus the same number of quiet ones.

    Each pool is shuffled once, independently of ``n_samples``, and the draw is a prefix of
    that shuffle -- so at a fixed seed a smaller draw is a strict subset of a larger one,
    which is what makes a 50/100/200 learning curve a nested sequence of training sets.

    If a split runs out of candidates the balance is still held and a warning reports the
    shortfall -- worth reading, because the active pool is the scarce one and a split can
    be far smaller than requested. Returns the selected rows with an ``is_event`` column
    (1 = active, 0 = quiet).
    """
    rng = np.random.default_rng(seed)
    act = active_candidates(matched, label_column, active_above)
    quiet = quiet_candidates(matched, label_column, quiet_below, active_above, non_event_buffer)
    # A quiet sample may not reuse a Surya frame already available as an active one.
    quiet = quiet[~quiet["valid_indices"].isin(act["valid_indices"])]

    act = act.iloc[rng.permutation(len(act))].reset_index(drop=True)
    quiet = quiet.iloc[rng.permutation(len(quiet))].reset_index(drop=True)

    n_active_wanted = len(act) if n_samples is None else n_samples // 2
    n_active = min(n_active_wanted, len(act))
    n_extra = 0 if n_samples is None or n_active < n_active_wanted else n_samples % 2
    n_quiet = min(n_active + n_extra, len(quiet))

    if n_active < n_active_wanted or n_quiet < n_active + n_extra:
        warnings.warn(
            f"Requested {n_samples} samples ({label_column} > {active_above} vs "
            f"< {quiet_below}) but only {len(act)} active and {len(quiet)} quiet frames "
            f"exist in this split; using {n_active} active + {n_quiet} quiet."
        )

    return pd.concat(
        [act.iloc[:n_active].assign(is_event=1), quiet.iloc[:n_quiet].assign(is_event=0)],
        ignore_index=True,
    )
