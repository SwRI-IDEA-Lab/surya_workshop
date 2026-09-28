"""Tests for the EUV2MAG app's config and its tokenizer channel mapping.

This app's config carries four channel lists that answer different questions, and mixing
them up is silent rather than loud: the run trains, the loss falls, and the tokenizer was
initialized from the wrong pretrained channels. These tests pin the checks that turn each
confusion into a config-load error, and the index mapping the tokenizer slice depends on.
"""

import copy

import pytest
import yaml

from downstream_apps.Imagetranslation.configs import (
    SURYA_PRETRAINED_CHANNEL_ORDER,
    Euv2MagDataConfig,
    load_euv2mag_config,
)

CONFIG = "downstream_apps/Imagetranslation/configs/config_script.yaml"


def make_data_config(**overrides) -> Euv2MagDataConfig:
    kwargs = dict(
        train_data_path="train.csv",
        valid_data_path="val.csv",
        scalers_path="scalers.yaml",
        channels=["aia171", "aia193", "aia304", "hmi_m"],
        time_delta_input_minutes=[0],
        time_delta_target_minutes=60,
    )
    kwargs.update(overrides)
    return Euv2MagDataConfig(**kwargs)


# ---------------------------------------------------------------------------
# The tokenizer channel mapping
# ---------------------------------------------------------------------------

def test_indices_are_positions_in_the_pretraining_order_not_in_channels():
    """The whole point of the separate list. data.channels here is
    [aia171, aia193, aia304, hmi_m], so aia304 is at position 2 in it -- and at position 5
    in the pretraining order, which is the one the checkpoint's convolution was built in.
    Reading the indices off data.channels would load three wrong channels' filters."""
    cfg = make_data_config(input_channels=["aia304", "aia193", "aia171"])
    assert cfg.input_channel_indices == [5, 3, 2]
    assert [cfg.channels.index(c) for c in cfg.input_channels] == [2, 1, 0]  # NOT this


def test_indices_follow_the_models_channel_order_not_ascending():
    """The order is the order the tokenizer expects its input channels in, so reordering
    input_channels must reorder the slice, not just relabel it."""
    assert make_data_config(
        input_channels=["aia171", "aia193", "aia304"]
    ).input_channel_indices == [2, 3, 5]


def test_pretraining_order_matches_the_checkpoints_channel_count():
    assert len(SURYA_PRETRAINED_CHANNEL_ORDER) == 13


def test_pretraining_order_is_not_the_scalers_yaml_order():
    """A regression guard on the trap the docstrings warn about: scalers.yaml lists
    hmi_bx/by/bz before hmi_m, so the two orders are genuinely different."""
    order = SURYA_PRETRAINED_CHANNEL_ORDER
    assert order.index("hmi_m") < order.index("hmi_bx")


# ---------------------------------------------------------------------------
# Channel-list validation
# ---------------------------------------------------------------------------

def test_a_target_that_is_also_an_input_raises():
    """Otherwise the model is handed the answer and the validation loss looks excellent."""
    with pytest.raises(ValueError, match="both contain"):
        make_data_config(input_channels=["aia171"], target_channels=["aia171"])


def test_an_input_channel_the_dataset_never_reads_raises():
    with pytest.raises(ValueError, match="data.channels does not contain"):
        make_data_config(input_channels=["aia94", "aia193", "aia171"])


def test_a_channel_listed_but_never_used_raises():
    """Every channel in data.channels is decoded for every sample, so a leftover costs
    real time on every batch."""
    with pytest.raises(ValueError, match="neither inputs nor targets"):
        make_data_config(channels=["aia171", "aia193", "aia304", "hmi_m", "hmi_v"])


def test_a_repeated_input_channel_raises():
    with pytest.raises(ValueError, match="repeats"):
        make_data_config(
            channels=["aia171", "aia193", "hmi_m"],
            input_channels=["aia171", "aia171", "aia193"],
        )


def test_an_empty_channel_list_raises():
    with pytest.raises(ValueError, match="must name at least one channel"):
        make_data_config(target_channels=[])


def test_an_input_channel_absent_from_the_pretraining_order_raises():
    """There are no pretrained tokenizer weights for it, so the slice cannot be built."""
    with pytest.raises(ValueError, match="pretrained_channel_order"):
        make_data_config(
            channels=["aia171", "made_up", "hmi_m"],
            input_channels=["aia171", "made_up"],
        )


# ---------------------------------------------------------------------------
# The cross-section check, via the real config file
# ---------------------------------------------------------------------------

def test_the_shipped_config_loads():
    cfg = load_euv2mag_config(CONFIG)
    assert cfg.model.in_channels == len(cfg.data.input_channels)
    assert cfg.model.img_size == cfg.data.native_img_size // cfg.data.pooling
    # The two settings the app depends on, pinned so a config edit cannot quietly undo them.
    assert cfg.model.trainable_backbone_modules == ["embedding.patch_embed"]
    assert cfg.drop_hmi_probability == 0.0, "would zero this app's prediction target"


def test_the_shipped_config_targets_the_backbones_real_attention_layers():
    """The bug this guards: the backbone uses a fused attn.qkv and attn.proj, PEFT only
    errors when *no* target matches, and the original config's q_proj/k_proj/v_proj/
    out_proj list silently adapted nothing but fc1/fc2."""
    targets = load_euv2mag_config(CONFIG).model.lora_config.target_modules
    assert "attn.qkv" in targets and "attn.proj" in targets
    assert not any(t in targets for t in ("q_proj", "k_proj", "v_proj", "out_proj"))
    # A bare "proj" would also match the tokenizer at embedding.patch_embed.proj.
    assert "proj" not in targets


def test_in_channels_disagreeing_with_input_channels_raises(tmp_path):
    raw = yaml.safe_load(open(CONFIG))
    raw["model"]["in_channels"] = 13  # the pretraining count, a tempting mistake
    path = tmp_path / "bad.yaml"
    # Paths resolve against the config file's directory, so absolute ones survive the move.
    raw["data"]["train_data_path"] = str(tmp_path / "t.csv")
    raw["data"]["valid_data_path"] = str(tmp_path / "v.csv")
    raw["data"]["scalers_path"] = str(tmp_path / "s.yaml")
    raw["model"]["pretrained_path"] = str(tmp_path / "w.pt")
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="does not match len\\(data.input_channels\\)"):
        load_euv2mag_config(str(path))
