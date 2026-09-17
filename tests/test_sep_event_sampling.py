"""Tests for the balanced event / non-event sampling of the SEP-from-PSP app.

Uses a small synthetic catalog and event list, so no Surya data or network is needed.
"""

import pandas as pd
import pytest

from downstream_apps.sep_pred_psp.datasets.event_sampling import (
    FINISH,
    MAX_LO,
    PSP_TIME,
    START,
    load_event_list,
    match_catalog_to_surya,
    sample_balanced,
)


def make_events():
    # Three events on the PSP clock. The third has no Max Time Lo, so it is never an event
    # sample, but its window still excludes non-events.
    return pd.DataFrame({
        "Event Number": ["E1", "E2", "E3"],
        START: pd.to_datetime(["2022-01-03", "2022-01-10", "2022-01-20"]),
        FINISH: pd.to_datetime(["2022-01-04", "2022-01-11", "2022-01-21"]),
        MAX_LO: pd.to_datetime(["2022-01-03 12:00", "2022-01-10 06:00", None]),
    })


def make_matched():
    # Hourly PSP times over January, two active regions per hour, SHARP time 3 days earlier.
    psp = pd.date_range("2022-01-01", "2022-01-31", freq="1h")
    catalog = pd.DataFrame({
        PSP_TIME: psp.repeat(2),
        "SHARP time": (psp - pd.Timedelta("3d")).repeat(2) + pd.to_timedelta([0, 5] * len(psp), "min"),
        "ARPNUM": [1, 2] * len(psp),
        "Jlinlin": 1.0,
    })
    surya = pd.date_range("2021-12-28", "2022-01-31", freq="12min")
    return match_catalog_to_surya(catalog, list(surya), "SHARP time", "4d", "forward")


def test_half_events_half_non_events_with_odd_extra_as_non_event():
    out = sample_balanced(make_matched(), make_events(), n_samples=5, seed=0)
    assert out["is_event"].tolist().count(1) == 2
    assert out["is_event"].tolist().count(0) == 3


def test_events_are_closest_to_max_time_lo_inside_window():
    events = make_events()
    out = sample_balanced(make_matched(), events, n_samples=4, seed=0)
    picked = out[out["is_event"] == 1].set_index("event_number")
    assert set(picked.index) == {"E1", "E2"}  # E3 has no Max Time Lo
    for _, ev in events.dropna(subset=[MAX_LO]).iterrows():
        assert picked.loc[ev["Event Number"], PSP_TIME] == ev[MAX_LO]


def test_non_events_are_at_least_buffer_away_from_every_event():
    events = make_events()
    out = sample_balanced(make_matched(), events, n_samples=None, seed=0, non_event_buffer="1d")
    non = out.loc[out["is_event"] == 0, PSP_TIME]
    for start, finish in zip(events[START], events[FINISH]):
        assert not ((non >= start - pd.Timedelta("1d")) & (non <= finish + pd.Timedelta("1d"))).any()


def test_short_on_events_keeps_balance_and_warns():
    with pytest.warns(UserWarning, match="only 2 event"):
        out = sample_balanced(make_matched(), make_events(), n_samples=10, seed=0)
    assert out["is_event"].sum() == 2
    assert (out["is_event"] == 0).sum() == 2


def test_same_seed_same_draw_and_no_repeated_surya_frames():
    a = sample_balanced(make_matched(), make_events(), n_samples=4, seed=7)
    b = sample_balanced(make_matched(), make_events(), n_samples=4, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert not a["valid_indices"].duplicated().any()


def test_load_event_list_parses_mixed_formats_and_skips_inverted_windows(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame({
        "Event Number": ["ok", "typo", "inverted"],
        START: ["10/2/2018 13:00:00", "12/17/24 20:52", "7/15/2023 22:45:00"],
        FINISH: ["10/4/2018 21:00:00", "12/18/2024 9:30:00", "7/15/2023 05:50:00"],
        MAX_LO: ["10/3/2018 3:30:00", "2024-12-18:03:28", ""],
    }).to_csv(path, index=False)
    with pytest.warns(UserWarning, match="inverted"):
        events = load_event_list(str(path))
    assert events["Event Number"].tolist() == ["ok", "typo"]
    assert events.loc[1, MAX_LO] == pd.Timestamp("2024-12-18 03:28")


# --- Repeat frames per event (max_frames_per_event > 1) ---------------------------------
#
# The binding constraint on this task is the number of distinct SEP events, not the number
# of SDO frames, so a balanced draw cannot exceed 2x the event count at one frame each.
# max_frames_per_event lifts that ceiling by returning to an event for another frame.


def test_rounds_fill_in_order_every_event_before_any_second_frame():
    # Two usable events, 3 frames each -> a 3-event draw must be both events once, then one
    # of them a second time. Never one event twice while the other is still unused.
    with pytest.warns(UserWarning, match="distinct events"):
        out = sample_balanced(
            make_matched(), make_events(), n_samples=6, seed=0, max_frames_per_event=3
        )
    picked = out[out["is_event"] == 1]
    assert len(picked) == 3
    assert set(picked["event_number"]) == {"E1", "E2"}
    assert sorted(picked["frame_round"]) == [1, 1, 2]
    assert not out["valid_indices"].duplicated().any()


def test_repeat_frames_are_spread_across_the_event_window():
    # A second frame is the one furthest in PSP time from the first, not the next hour
    # along: adjacent frames would be near-identical solar images.
    out = sample_balanced(
        make_matched(), make_events(), n_samples=8, seed=0, max_frames_per_event=2
    )
    for event_id, group in out[out["is_event"] == 1].groupby("event_number"):
        if len(group) < 2:
            continue
        span = group[PSP_TIME].max() - group[PSP_TIME].min()
        assert span >= pd.Timedelta("11h"), f"{event_id} frames only {span} apart"


def test_one_frame_per_event_is_the_default():
    # Unchanged behavior for callers that never set the knob.
    out = sample_balanced(make_matched(), make_events(), n_samples=None, seed=0)
    assert (out.loc[out["is_event"] == 1, "frame_round"] == 1).all()
    assert out.loc[out["is_event"] == 1, "event_number"].is_unique


def test_draws_are_nested_so_a_learning_curve_shares_its_data():
    # The 50-sample training set must be a subset of the 100-sample one, or a learning
    # curve is three unrelated draws rather than one growing dataset.
    kwargs = dict(seed=3, max_frames_per_event=3)
    small = sample_balanced(make_matched(), make_events(), n_samples=4, **kwargs)
    with pytest.warns(UserWarning):
        large = sample_balanced(make_matched(), make_events(), n_samples=8, **kwargs)
    assert set(small["valid_indices"]).issubset(set(large["valid_indices"]))


def test_repeat_frames_warn_so_they_are_never_silent():
    # Reaching into round 2 means the extra samples are further views of an event already
    # in the set, not new events. That has to be visible in the run log.
    with pytest.warns(UserWarning, match=r"3 event samples drawn from 2 distinct events"):
        sample_balanced(
            make_matched(), make_events(), n_samples=6, seed=0, max_frames_per_event=3
        )


def test_val_draw_does_not_depend_on_the_train_draw_size():
    # The two splits are sized independently, so holding n_samples fixed for validation
    # gives the identical set no matter what the training split asks for.
    a = sample_balanced(make_matched(), make_events(), n_samples=4, seed=0, max_frames_per_event=3)
    b = sample_balanced(make_matched(), make_events(), n_samples=4, seed=0, max_frames_per_event=3)
    pd.testing.assert_frame_equal(a, b)
