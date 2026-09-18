"""Tests for the balanced active / quiet sampling of the SEP-from-PSP app.

Uses a small synthetic catalog, so no Surya data, event list or network is needed.
"""

import numpy as np
import pandas as pd
import pytest

from downstream_apps.sep_pred_psp.datasets.event_sampling import (
    PSP_TIME,
    match_catalog_to_surya,
    sample_balanced,
)

LABEL = "Jlinlin"
ACTIVE_ABOVE, QUIET_BELOW = 1.0, 0.1


def make_matched(active_hours: int = 48):
    """Hourly PSP times over January, two active regions per hour, SHARP time 3 days back.

    The first ``active_hours`` hours of 2022-01-10 onward are far above ``ACTIVE_ABOVE``;
    a middle band sits between the thresholds; everything else is deep below
    ``QUIET_BELOW``.
    """
    psp = pd.date_range("2022-01-01", "2022-01-31", freq="1h")
    label = pd.Series(0.01, index=range(len(psp)))
    active_start = psp.get_loc(pd.Timestamp("2022-01-10"))
    label.iloc[active_start:active_start + active_hours] = 5.0
    # A band between the thresholds: never sampled, on either side.
    label.iloc[:24] = 0.5

    catalog = pd.DataFrame({
        PSP_TIME: psp.repeat(2),
        "SHARP time": (psp - pd.Timedelta("3d")).repeat(2)
        + pd.to_timedelta([0, 5] * len(psp), "min"),
        "ARPNUM": [1, 2] * len(psp),
        LABEL: label.repeat(2).to_numpy(),
    })
    surya = pd.date_range("2021-12-28", "2022-01-31", freq="12min")
    return match_catalog_to_surya(catalog, list(surya), "SHARP time", "4d", "forward")


def draw(matched=None, *, n_samples, seed=0, buffer="1d"):
    return sample_balanced(
        matched if matched is not None else make_matched(),
        n_samples=n_samples,
        seed=seed,
        label_column=LABEL,
        active_above=ACTIVE_ABOVE,
        quiet_below=QUIET_BELOW,
        non_event_buffer=buffer,
    )


def test_half_active_half_quiet_with_odd_extra_as_quiet():
    out = draw(n_samples=5)
    assert out["is_event"].tolist().count(1) == 2
    assert out["is_event"].tolist().count(0) == 3


def test_classes_obey_the_thresholds():
    out = draw(n_samples=20)
    assert (out.loc[out["is_event"] == 1, LABEL] > ACTIVE_ABOVE).all()
    assert (out.loc[out["is_event"] == 0, LABEL] < QUIET_BELOW).all()


def test_the_band_between_the_thresholds_is_never_sampled():
    out = draw(n_samples=40)
    between = out[(out[LABEL] >= QUIET_BELOW) & (out[LABEL] <= ACTIVE_ABOVE)]
    assert between.empty, "a frame from the dropped band was sampled"


def test_quiet_samples_are_at_least_buffer_away_from_every_active_hour():
    matched = make_matched()
    out = draw(matched, n_samples=30, buffer="1d")
    active_times = matched.loc[matched[LABEL] > ACTIVE_ABOVE, PSP_TIME].to_numpy()
    for t in out.loc[out["is_event"] == 0, PSP_TIME]:
        gap = np.abs(active_times - t.to_datetime64()).min()
        assert gap >= pd.Timedelta("1d").to_timedelta64()


def test_one_sample_per_surya_frame():
    out = draw(n_samples=60)
    assert out["valid_indices"].is_unique


def test_draws_are_nested_so_a_learning_curve_grows_one_dataset():
    small = draw(n_samples=10, seed=7)
    large = draw(n_samples=40, seed=7)
    assert set(small["valid_indices"]) <= set(large["valid_indices"])


def test_a_different_seed_gives_a_different_draw():
    a = draw(n_samples=20, seed=1)
    b = draw(n_samples=20, seed=2)
    assert set(a["valid_indices"]) != set(b["valid_indices"])


def test_none_uses_every_active_candidate():
    matched = make_matched()
    out = draw(matched, n_samples=None)
    n_active = matched[matched[LABEL] > ACTIVE_ABOVE]["valid_indices"].nunique()
    assert int(out["is_event"].sum()) == n_active


def test_a_split_short_on_active_frames_warns_and_stays_balanced():
    # The scarce class caps the draw: 2 active hours can never fill 50 samples.
    matched = make_matched(active_hours=2)
    with pytest.warns(UserWarning, match="only .* active"):
        out = draw(matched, n_samples=50)
    n_active = int(out["is_event"].sum())
    assert n_active == int((out["is_event"] == 0).sum()) <= 4
