"""Tests for per-split dataset sizing in ``build_helio_datasets()``.

A learning curve needs the training split to grow while the validation split stays put.
That is only possible if the builders can pass different task kwargs to each split, which
is what ``train_kwargs`` / ``val_kwargs`` are for. Uses a stub dataset class, so no Surya
data, scalers or network are needed.
"""

from types import SimpleNamespace

import pytest

from workshop_infrastructure.datasets.builders import build_helio_datasets


class RecordingDataset:
    """Stands in for a HelioNetCDFDataset subclass; just records how it was constructed."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __len__(self):
        return 0


def make_cfg():
    return SimpleNamespace(
        data=SimpleNamespace(
            train_data_path="train.csv",
            valid_data_path="val.csv",
            scalers_path="scalers.yaml",
            channels=["aia94"],
            time_delta_input_minutes=[0],
            time_delta_target_minutes=60,
            sdo_data_root_path=None,
            s3_mode="download",
            s3_anon=True,
            s3_cache_dir=None,
            s3_boto3_max_concurrency=4,
            s3_boto3_part_size_mb=64,
        ),
        model=SimpleNamespace(time_embedding=SimpleNamespace(time_dim=1)),
        rollout_steps=0,
        drop_hmi_probability=0.0,
        use_latitude_in_learned_flow=False,
    )


def build(**kwargs):
    return build_helio_datasets(make_cfg(), RecordingDataset, scalers={}, **kwargs)


def test_split_specific_kwargs_reach_only_their_own_split():
    train, val = build(
        train_kwargs={"max_number_of_samples": 500},
        val_kwargs={"max_number_of_samples": 50},
    )
    assert train.kwargs["max_number_of_samples"] == 500
    assert val.kwargs["max_number_of_samples"] == 50
    assert train.kwargs["phase"] == "train" and val.kwargs["phase"] == "val"
    assert train.kwargs["index_path"] == "train.csv"
    assert val.kwargs["index_path"] == "val.csv"


def test_shared_task_kwargs_still_reach_both_splits():
    train, val = build(
        active_above=1.0,
        train_kwargs={"max_number_of_samples": 500},
        val_kwargs={"max_number_of_samples": 50},
    )
    assert train.kwargs["active_above"] == 1.0
    assert val.kwargs["active_above"] == 1.0


def test_split_kwargs_override_a_shared_one_rather_than_raising():
    # Two ** expansions of dicts sharing a key is a TypeError, not an override; the
    # builders merge first so that an app can set a default and override it for one split.
    train, val = build(max_number_of_samples=10, val_kwargs={"max_number_of_samples": 50})
    assert train.kwargs["max_number_of_samples"] == 10
    assert val.kwargs["max_number_of_samples"] == 50


def test_omitting_both_keeps_the_splits_identical():
    train, val = build(max_number_of_samples=10)
    assert train.kwargs["max_number_of_samples"] == val.kwargs["max_number_of_samples"] == 10
