"""
Balanced event / non-event sample selection for the SEP-from-PSP task.

The SHARP-to-PSP catalog has one row per (PSP hour, active region). Each row carries two
clocks: ``time`` (when the particles arrive at PSP) and ``SHARP time`` (the solar-surface
time about 3 days earlier, used to pick the Surya frame). The ISOIS event list is on the
PSP clock, so event membership is always decided on ``time``.

Selection, per data split:

1. Match every catalog row to the Surya frame at ``SHARP time`` (within a tolerance).
2. **Event** candidates: up to ``max_frames_per_event`` rows per event, drawn in *rounds*.
   Round 1 is the row whose PSP ``time`` is closest to the event's ``Max Time Lo``, inside
   its Start-Finish window; round 2 adds the row furthest in PSP time from that one, and so
   on. Events with no Max Time Lo, or with Max Time Lo outside the window, are skipped.
3. **Non-event** candidates: rows whose PSP ``time`` is at least ``non_event_buffer`` away
   from every event window.
4. Draw N//2 events and the rest non-events at random (seeded). Event rounds are
   consumed in order, so every event contributes one frame before any event contributes a
   second — a bigger N reaches for a new event before it reuses one.

Everything here is a pure function of DataFrames, so it can be tested without Surya data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

PSP_TIME = "time"
START, FINISH, MAX_LO = "Start (UTC)", "Finish (UTC)", "Max Time Lo (UTC)"
EVENT_ID = "Event Number"

# Catalog-side direction for pd.merge_asof. ds_match_direction is phrased from the Surya
# side ("forward": the catalog time is at or after the Surya frame, i.e. causal), while
# here the catalog is the left table, so each direction flips.
_CATALOG_DIRECTION = {"forward": "backward", "backward": "forward", "nearest": "nearest"}


def _parse_times(series: pd.Series) -> pd.Series:
    """Parse the event list's hand-entered timestamps.

    The file mixes formats ("10/2/2018 13:00:00", "12/17/24 20:52", "2023-01-02 7:24") and
    has at least one "2023-01-04:11:09" typo, so a single fixed format does not work.
    """
    cleaned = series.astype("string").str.strip().str.replace(
        r"^(\d{4}-\d{2}-\d{2}):", r"\1 ", regex=True
    )
    return pd.to_datetime(cleaned, format="mixed", errors="coerce")


def load_event_list(path: str) -> pd.DataFrame:
    """Load the ISOIS event list with parsed Start/Finish/Max Time Lo.

    Rows whose Start or Finish is missing, or whose Finish is before Start, are dropped with
    a warning: their window is unknown, so they can be used neither as events nor to decide
    what counts as "outside all events". Fix them in the CSV to bring them back.
    """
    events = pd.read_csv(path)
    for col in (START, FINISH, MAX_LO):
        events[col] = _parse_times(events[col])

    bad = events[START].isna() | events[FINISH].isna() | (events[FINISH] < events[START])
    if bad.any():
        warnings.warn(
            f"Skipping {int(bad.sum())} event(s) with a missing or inverted Start/Finish "
            f"window: {events.loc[bad, EVENT_ID].tolist()}"
        )
    return events.loc[~bad].reset_index(drop=True)


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


def _pick_frames(in_window: pd.DataFrame, max_lo, k: int, rng) -> list:
    """Up to ``k`` rows from one event window, each far in PSP time from the ones before it.

    The first pick is the row closest to ``Max Time Lo`` (ties broken at random), so ``k=1``
    is the single most representative frame for the event. Each later pick maximizes the
    gap to everything already picked, which spreads repeat frames across the window instead
    of taking the next hour along — adjacent frames would be near-identical solar images.
    """
    # One row per Surya frame: several active regions can map to the same frame, and the
    # closest SHARP match is the one to keep.
    pool = in_window.sort_values("index_delta", kind="stable").drop_duplicates("valid_indices")
    gap = (pool[PSP_TIME] - max_lo).abs()
    closest = pool[gap == gap.min()]
    first = closest.iloc[rng.integers(len(closest))]

    picks = [first]
    pool = pool.drop(index=first.name)
    while len(picks) < k and len(pool) > 0:
        taken = np.array([p[PSP_TIME].to_datetime64() for p in picks])
        distance = np.abs(pool[PSP_TIME].to_numpy()[:, None] - taken[None, :]).min(axis=1)
        nxt = pool.iloc[int(distance.argmax())]
        picks.append(nxt)
        pool = pool.drop(index=nxt.name)
    return picks


def event_candidates(
    matched: pd.DataFrame,
    events: pd.DataFrame,
    rng,
    max_frames_per_event: int = 1,
) -> pd.DataFrame:
    """Up to ``max_frames_per_event`` catalog rows per event, ordered in rounds.

    The ``frame_round`` column records which round a row came from: round 1 holds one row
    per event (the frame closest to Max Time Lo), round 2 a second frame per event, and so
    on. Rows come back sorted by round, so a caller that takes a prefix exhausts every
    event before reusing one. A Surya frame belongs to a single sample — where two events
    would claim the same frame, the earlier round keeps it.
    """
    usable = events.dropna(subset=[MAX_LO])
    usable = usable[(usable[MAX_LO] >= usable[START]) & (usable[MAX_LO] <= usable[FINISH])]

    rows = []
    for _, event in usable.iterrows():
        in_window = matched[
            (matched[PSP_TIME] >= event[START]) & (matched[PSP_TIME] <= event[FINISH])
        ]
        if in_window.empty:
            continue  # no Surya frame for this event in this split
        for round_number, row in enumerate(
            _pick_frames(in_window, event[MAX_LO], max_frames_per_event, rng), start=1
        ):
            row = row.copy()
            row["event_number"] = event[EVENT_ID]
            row["frame_round"] = round_number
            rows.append(row)

    columns = list(matched.columns) + ["event_number", "frame_round"]
    out = pd.DataFrame(rows, columns=columns)
    out = out.sort_values("frame_round", kind="stable").drop_duplicates("valid_indices")
    return out.reset_index(drop=True)


def non_event_candidates(
    matched: pd.DataFrame, events: pd.DataFrame, buffer: str = "1d"
) -> pd.DataFrame:
    """Rows whose PSP ``time`` is at least ``buffer`` away from every event window."""
    pad = pd.Timedelta(buffer)
    psp = matched[PSP_TIME].to_numpy()
    near_event = np.zeros(len(matched), dtype=bool)
    for start, finish in zip(events[START] - pad, events[FINISH] + pad):
        near_event |= (psp >= start.to_datetime64()) & (psp <= finish.to_datetime64())
    out = matched.loc[~near_event].copy()
    out["event_number"] = ""
    out["frame_round"] = 0  # 0 = not drawn from an event window
    return out.reset_index(drop=True)


def sample_balanced(
    matched: pd.DataFrame,
    events: pd.DataFrame,
    n_samples: int | None,
    seed: int,
    non_event_buffer: str = "1d",
    max_frames_per_event: int = 1,
) -> pd.DataFrame:
    """Draw a half-event / half-non-event sample, one row per Surya frame.

    ``n_samples`` gives N//2 events and N - N//2 non-events (an odd extra is a non-event).
    ``n_samples=None`` uses every event candidate plus the same number of non-events.

    ``max_frames_per_event`` is the ceiling on how many Surya frames one event may
    contribute. With the default of 1 the event samples are one per event and fully
    independent; raising it lets the draw grow past the number of distinct events in the
    split by taking a second (then third, ...) frame from each event window, spread as far
    apart in PSP time as the window allows. Those extra frames are repeat views of an event
    already in the set, not new events, so the draw warns when it reaches into round 2.

    Rounds are consumed in order and every pool is shuffled once per round, independently
    of ``n_samples`` — so at a fixed seed the draw for a smaller ``n_samples`` is a strict
    prefix of the draw for a larger one. That is what makes a 50/100/200 learning curve a
    nested sequence of training sets rather than three unrelated draws.

    If the split runs out of candidates the balance is still held and a warning reports the
    shortfall. Returns the selected rows with ``is_event`` (0/1), ``event_number`` and
    ``frame_round`` columns.
    """
    rng = np.random.default_rng(seed)
    ev = event_candidates(matched, events, rng, max_frames_per_event)
    # A non-event may not reuse a Surya frame already chosen as an event.
    non = non_event_candidates(matched, events, non_event_buffer)
    non = non[~non["valid_indices"].isin(ev["valid_indices"])].drop_duplicates("valid_indices")

    # Shuffle inside each round, then keep the rounds in order: a larger draw reaches for
    # an unused event before it takes a second frame from one it already has.
    if len(ev):
        ev = pd.concat(
            [g.iloc[rng.permutation(len(g))] for _, g in ev.groupby("frame_round", sort=True)],
            ignore_index=True,
        )
    non = non.iloc[rng.permutation(len(non))].reset_index(drop=True)

    n_event_wanted = len(ev) if n_samples is None else n_samples // 2
    n_event = min(n_event_wanted, len(ev))
    n_extra = 0 if n_samples is None or n_event < n_event_wanted else n_samples % 2
    n_non = min(n_event + n_extra, len(non))

    if n_event < n_event_wanted or n_non < n_event + n_extra:
        warnings.warn(
            f"Requested {n_samples} samples but only {len(ev)} event and {len(non)} "
            f"non-event candidates exist; using {n_event} events + {n_non} non-events."
        )

    picked_ev = ev.iloc[:n_event].assign(is_event=1)
    picked_non = non.iloc[:n_non].assign(is_event=0)

    if n_event and int(picked_ev["frame_round"].max()) > 1:
        distinct = picked_ev["event_number"].nunique()
        warnings.warn(
            f"{n_event} event samples drawn from {distinct} distinct events "
            f"(up to {int(picked_ev['frame_round'].max())} frames per event). The "
            f"{n_event - distinct} samples beyond the first {distinct} are further frames "
            f"from an event already in the set, not new events."
        )

    return pd.concat([picked_ev, picked_non], ignore_index=True)
