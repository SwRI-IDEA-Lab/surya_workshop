"""Tests for the workshop-facing defaults and the app-local fine-tuning model.

Three things are pinned here:

* the config-load-time validations that turn silent footguns into named errors
  (``model.img_size`` vs ``data.pooling``, the retired ``training.dtype`` key,
  ``precision``, per-split sample caps);
* ``HelioNetCDFDataset._apply_subsample``'s two properties — random, and *nested* across
  sizes — since a data-scaling curve is meaningless without them;
* ``FlareSuryaModel``, the model students edit, agreeing with the infrastructure model it
  replaces and staying compatible with the ``head_`` LoRA convention.
"""

import numpy as np
import pytest
import torch
import yaml

from downstream_apps.template.configs import load_flare_config
from downstream_apps.template.models.finetune_model import FlareSuryaModel
from workshop_infrastructure.datasets.helio import HelioNetCDFDataset
from workshop_infrastructure.utils import apply_peft_lora, discover_head_modules

CONFIG_PATH = "downstream_apps/template/configs/config_script.yaml"


# ---------------------------------------------------------------------------
# The shipped config
# ---------------------------------------------------------------------------

def test_shipped_config_is_the_low_memory_default():
    cfg = load_flare_config(CONFIG_PATH)
    # 1024x1024 frames: 4096 tokens instead of 65536, which is what makes a usable
    # batch size fit in 40 GB.
    assert cfg.data.pooling == 4
    assert cfg.model.img_size == cfg.data.native_img_size // cfg.data.pooling == 1024
    assert cfg.model.pooling == "global_average"
    assert cfg.precision == "bf16-mixed"
    # Train and validation are capped independently.
    assert cfg.data.max_train_samples != cfg.data.max_val_samples


def _config_with(tmp_path, **overrides):
    """Write a copy of the shipped config with nested keys overridden ('a.b' syntax)."""
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    for dotted, value in overrides.items():
        section, key = dotted.split(".")
        if value is None:
            raw[section].pop(key, None)
        else:
            raw[section][key] = value
    path = tmp_path / "cfg.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(raw, f)
    return path


def test_img_size_disagreeing_with_pooling_is_rejected_at_load_time(tmp_path):
    # Reducing pooling without reducing img_size used to start a run that failed inside
    # the first forward pass with an opaque reshape error.
    path = _config_with(tmp_path, **{"data.pooling": 2})
    with pytest.raises(ValueError, match="model.img_size .* does not match"):
        load_flare_config(path)


def test_retired_dtype_key_names_its_replacement(tmp_path):
    # training.dtype was accepted and had no effect whatsoever; it now errors, and the
    # message has to point at training.precision rather than list 15 valid keys.
    path = _config_with(tmp_path, **{"training.dtype": "bfloat16"})
    with pytest.raises(ValueError, match="training.precision"):
        load_flare_config(path)


def test_unknown_precision_is_rejected(tmp_path):
    path = _config_with(tmp_path, **{"training.precision": "bf16"})
    with pytest.raises(ValueError, match="training.precision"):
        load_flare_config(path)


def test_max_samples_is_a_shorthand_for_both_caps(tmp_path):
    path = _config_with(
        tmp_path,
        **{"data.max_samples": 7, "data.max_train_samples": None, "data.max_val_samples": None},
    )
    cfg = load_flare_config(path)
    assert cfg.data.max_train_samples == 7
    assert cfg.data.max_val_samples == 7


def test_explicit_cap_wins_over_the_shorthand(tmp_path):
    path = _config_with(
        tmp_path, **{"data.max_samples": 7, "data.max_train_samples": 3, "data.max_val_samples": None}
    )
    cfg = load_flare_config(path)
    assert cfg.data.max_train_samples == 3
    assert cfg.data.max_val_samples == 7


def test_new_training_keys_need_no_second_registration(tmp_path):
    """The training: key list is derived from TrainingConfig's fields.

    It used to be a hand-maintained frozenset, so adding a field without updating it made
    the YAML key an error and updating only the set made it a silent no-op.
    """
    from dataclasses import fields

    from workshop_infrastructure.configs import _TRAINING_KEYS, TrainingConfig

    declared = {f.name for f in fields(TrainingConfig)} - {"job_id", "data", "model", "output"}
    assert _TRAINING_KEYS | {"wandb_project", "wandb_entity"} == declared


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

class _FakeDataset(HelioNetCDFDataset):
    """Exercises _apply_subsample alone, without touching NetCDF, S3 or scalers."""

    def __init__(self, n, max_number_of_samples=None, subsample_seed=42):
        self.valid_indices = list(range(n))
        self.adjusted_length = n
        self.max_number_of_samples = max_number_of_samples
        self.subsample_seed = subsample_seed


def test_subsample_is_random_not_the_head_of_the_index():
    ds = _FakeDataset(1000, max_number_of_samples=20)
    ds._apply_subsample()
    assert len(ds.valid_indices) == 20
    # A chronological truncation would give exactly range(20).
    assert ds.valid_indices != list(range(20))
    # Still ordered, just sparse.
    assert ds.valid_indices == sorted(ds.valid_indices)
    assert max(ds.valid_indices) > 100


def test_subsets_nest_so_a_scaling_curve_measures_more_data_not_different_data():
    small = _FakeDataset(1000, max_number_of_samples=50)
    large = _FakeDataset(1000, max_number_of_samples=200)
    small._apply_subsample()
    large._apply_subsample()
    assert set(small.valid_indices) < set(large.valid_indices)


def test_subsample_returns_positions_so_parallel_frames_stay_aligned():
    ds = _FakeDataset(100, max_number_of_samples=10)
    keep = ds._apply_subsample()
    assert isinstance(keep, np.ndarray) and len(keep) == 10
    # valid_indices was range(100), so the kept values are the kept positions.
    assert ds.valid_indices == keep.tolist()


def test_no_cap_leaves_the_index_untouched():
    ds = _FakeDataset(10, max_number_of_samples=None)
    assert ds._apply_subsample() is None
    assert ds.adjusted_length == 10
    # A cap larger than the dataset is also a no-op.
    ds = _FakeDataset(10, max_number_of_samples=50)
    assert ds._apply_subsample() is None
    assert ds.adjusted_length == 10


def test_flare_dataset_defers_subsampling_until_after_the_label_join():
    """Subsampling in the base __init__ would cap the *timesteps*, so which flares got
    matched would depend on the cap. The subclass opts out and calls it itself."""
    from downstream_apps.template.datasets.template_dataset import FlareDSDataset

    assert FlareDSDataset.SUBSAMPLE_IN_BASE_INIT is False
    assert HelioNetCDFDataset.SUBSAMPLE_IN_BASE_INIT is True


# ---------------------------------------------------------------------------
# The app-local model
# ---------------------------------------------------------------------------

def _tiny_model_cfg():
    from workshop_infrastructure.configs import ModelConfig, TimeEmbeddingConfig

    return ModelConfig(
        img_size=64,
        patch_size=16,
        in_channels=13,
        embed_dim=32,
        depth=3,
        spectral_blocks=1,
        num_heads=2,
        mlp_ratio=4.0,
        drop_rate=0.0,
        window_size=2,
        dp_rank=2,
        dropout=0.0,
        pooling="global_average",
        penultimate_linear_layer=True,
        checkpoint_layers=[],
        time_embedding=TimeEmbeddingConfig(type="linear", time_dim=1),
    )


def _tiny_batch(batch_size=2):
    return {
        "ts": torch.randn(batch_size, 13, 1, 64, 64),
        "time_delta_input": torch.zeros(batch_size, 1),
    }


def test_app_model_follows_the_head_naming_convention():
    model = FlareSuryaModel(_tiny_model_cfg())
    # discover_head_modules raises if any trainable direct child is not head_*.
    assert set(discover_head_modules(model)) == {"head_linear", "head_unembed"}


def test_app_model_forward_returns_one_prediction_per_sample():
    model = FlareSuryaModel(_tiny_model_cfg()).eval()
    with torch.no_grad():
        out = model(_tiny_batch(batch_size=3))
    assert out.shape == (3,)


def test_app_model_head_stays_trainable_under_lora():
    from workshop_infrastructure.configs import LoraAdapterConfig

    model = FlareSuryaModel(_tiny_model_cfg())
    model = apply_peft_lora(model, LoraAdapterConfig())
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any("head_unembed" in n for n in trainable)
    assert any("head_linear" in n for n in trainable)
    assert any("lora_" in n for n in trainable)


def test_pooling_before_the_linear_layer_is_the_same_function():
    """global_average pools first because mean() and an affine layer commute.

    HelioSpectformer1D still applies head_linear to every token for the other poolings,
    so this equivalence is what licenses the shortcut -- if it ever stops holding, the
    optimization is silently changing the model.
    """
    torch.manual_seed(0)
    linear = torch.nn.Linear(32, 32)
    tokens = torch.randn(4, 100, 32)
    assert torch.allclose(linear(tokens).mean(dim=1), linear(tokens.mean(dim=1)), atol=1e-5)


@pytest.mark.parametrize("pooling", ["global_average", "global_max"])
def test_app_model_supports_both_its_documented_poolings(pooling):
    cfg = _tiny_model_cfg()
    cfg.pooling = pooling
    model = FlareSuryaModel(cfg).eval()
    with torch.no_grad():
        assert model(_tiny_batch(batch_size=2)).shape == (2,)


def test_app_model_refuses_a_pooling_it_does_not_implement():
    """Reading model.pooling and then ignoring it would be worse than refusing: the
    student edits the YAML, nothing changes, and nothing says why."""
    cfg = _tiny_model_cfg()
    cfg.pooling = "class_token"
    with pytest.raises(ValueError, match="HelioSpectformer1D"):
        FlareSuryaModel(cfg)


@pytest.mark.parametrize("pooling", ["global_average", "global_max"])
def test_app_model_matches_the_infrastructure_model_it_replaces(pooling):
    """FlareSuryaModel is the editable copy of HelioSpectformer1D's global_average path.

    Same weights in, same predictions out -- so a student who edits the app model is
    starting from the behaviour the template documents, not from a subtle variant.
    """
    from workshop_infrastructure.models.finetune_models import HelioSpectformer1D

    cfg = _tiny_model_cfg()
    cfg.pooling = pooling
    torch.manual_seed(0)
    app = FlareSuryaModel(cfg).eval()
    infra = HelioSpectformer1D.from_config(cfg, num_outputs=1).eval()
    infra.load_state_dict(app.state_dict())

    batch = _tiny_batch(batch_size=2)
    with torch.no_grad():
        assert torch.allclose(app(batch), infra(batch), atol=1e-5)
