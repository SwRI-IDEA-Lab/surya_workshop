"""Tests for the config-level cross-field validation and the nglo/pooling fix.

Guards the bug where ``model.nglo`` had to be hand-kept in sync with ``model.pooling`` or
the long-short attention reshape failed deep in vendored code, on the first forward pass
(not at construction). ``nglo`` is now derived from ``pooling`` inside
``HelioSpectformer1D`` rather than being a config field, so this also guards the other
config-level invariants added alongside it: ``ModelConfig.__post_init__`` and the
cross-section checks in ``TrainingConfig.__post_init__``.

Everything runs on CPU with a tiny backbone, so the suite is fast.
"""

import dataclasses

import pytest

from conftest import make_batch, make_model
from downstream_apps.template.configs import load_flare_config
from workshop_infrastructure.configs import (
    DataConfig,
    ModelConfig,
    TimeEmbeddingConfig,
    TrainingConfig,
)


def make_data_config(**overrides):
    kwargs = dict(
        train_data_path="train.csv",
        valid_data_path="valid.csv",
        scalers_path="scalers.yaml",
        channels=["aia171"],
        time_delta_input_minutes=[0],
        time_delta_target_minutes=60,
    )
    kwargs.update(overrides)
    return DataConfig(**kwargs)


def make_training_config(**overrides):
    kwargs = dict(job_id="test", data=make_data_config(), model=ModelConfig())
    kwargs.update(overrides)
    return TrainingConfig(**kwargs)


# ---------------------------------------------------------------------------
# nglo is derived, not configured
# ---------------------------------------------------------------------------


def test_model_config_has_no_nglo_field():
    assert "nglo" not in {f.name for f in dataclasses.fields(ModelConfig)}


def test_shipped_config_loads_and_has_no_nglo():
    cfg = load_flare_config("downstream_apps/template/configs/config_script.yaml")
    assert not hasattr(cfg.model, "nglo")


@pytest.mark.parametrize("pooling", ["class_token", "transformer", "attention", "global_average"])
def test_every_pooling_runs_a_forward_pass_with_the_derived_nglo(pooling):
    """The original bug only surfaced in forward(), not at construction."""
    model = make_model(pooling=pooling)
    output = model(make_batch())
    assert output.shape == (2,)


# ---------------------------------------------------------------------------
# ModelConfig.__post_init__
# ---------------------------------------------------------------------------


def test_default_model_config_is_valid():
    ModelConfig()  # must not raise


def test_img_size_not_divisible_by_patch_size_raises():
    with pytest.raises(ValueError, match="img_size"):
        ModelConfig(img_size=100, patch_size=16)


def test_img_size_divisible_by_patch_size_is_accepted():
    ModelConfig(img_size=64, patch_size=16)


def test_spectral_blocks_greater_than_depth_raises():
    with pytest.raises(ValueError, match="spectral_blocks"):
        ModelConfig(depth=3, spectral_blocks=4)


@pytest.mark.parametrize("spectral_blocks", [0, 3])
def test_spectral_blocks_at_the_boundary_is_accepted(spectral_blocks):
    ModelConfig(depth=3, spectral_blocks=spectral_blocks, checkpoint_layers=[])


def test_checkpoint_layers_out_of_range_raises():
    with pytest.raises(ValueError, match="checkpoint_layers"):
        ModelConfig(depth=3, checkpoint_layers=[0, 3])


def test_checkpoint_layers_negative_raises():
    with pytest.raises(ValueError, match="checkpoint_layers"):
        ModelConfig(depth=3, checkpoint_layers=[-1])


def test_checkpoint_layers_in_range_is_accepted():
    ModelConfig(depth=3, checkpoint_layers=[0, 1, 2])


def test_learned_flow_with_non_linear_time_embedding_raises():
    with pytest.raises(ValueError, match="learned_flow"):
        ModelConfig(learned_flow=True, time_embedding=TimeEmbeddingConfig(type="perceiver"))


def test_learned_flow_with_linear_time_embedding_is_accepted():
    ModelConfig(learned_flow=True, time_embedding=TimeEmbeddingConfig(type="linear"))


# ---------------------------------------------------------------------------
# TrainingConfig.__post_init__ (cross-section checks)
# ---------------------------------------------------------------------------


def test_deterministic_true_with_learned_flow_raises():
    with pytest.raises(ValueError, match="learned_flow"):
        make_training_config(
            model=ModelConfig(learned_flow=True), deterministic=True,
        )


def test_deterministic_warn_with_learned_flow_is_accepted():
    make_training_config(model=ModelConfig(learned_flow=True), deterministic="warn")


def test_time_dim_greater_than_available_deltas_raises():
    with pytest.raises(ValueError, match="time_dim"):
        make_training_config(
            data=make_data_config(time_delta_input_minutes=[0, -60]),
            model=ModelConfig(time_embedding=TimeEmbeddingConfig(time_dim=3)),
        )


def test_time_dim_less_than_or_equal_to_available_deltas_is_accepted():
    make_training_config(
        data=make_data_config(time_delta_input_minutes=[0, -60]),
        model=ModelConfig(time_embedding=TimeEmbeddingConfig(time_dim=2)),
    )
