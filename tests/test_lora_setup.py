"""Tests for the LoRA fine-tuning setup.

These guard two bugs that made every ``use_lora: true`` run meaningless:

1. The fine-tuning head was frozen at its random initialisation, because
   ``apply_peft_lora()`` passed no ``modules_to_save``.
2. ``target_modules`` named layers (``q_proj``/``k_proj``/``v_proj``/
   ``out_proj``) that do not exist in the Surya backbone, which uses a fused
   ``attn.qkv``.  PEFT only errors when *no* entry matches, so the attention
   layers were silently never adapted.

Everything runs on CPU with a tiny backbone, so the suite is fast.
"""

import pytest
import torch
from torch import nn

from conftest import DEPTH, EMBED_DIM, N_ATTENTION_BLOCKS, N_SPECTRAL_BLOCKS, make_batch, make_model
from workshop_infrastructure.configs import LoraAdapterConfig
from workshop_infrastructure.models.finetune_models import ClassToken
from workshop_infrastructure.utils import HEAD_PREFIX, apply_peft_lora, discover_head_modules


def adapted_modules(peft_model):
    """Qualified names of the modules PEFT wrapped with a LoRA adapter."""
    return {
        name.split(".lora_A")[0].replace("base_model.model.", "")
        for name, _ in peft_model.named_parameters()
        if ".lora_A" in name
    }


# ---------------------------------------------------------------------------
# target_modules
# ---------------------------------------------------------------------------


def test_adapted_modules_are_exactly_the_intended_set():
    """fc1/fc2 in every block, plus attn.qkv/attn.proj in the attention blocks."""
    model = apply_peft_lora(make_model(), LoraAdapterConfig())

    expected = set()
    for i in range(N_SPECTRAL_BLOCKS):
        prefix = f"backbone.backbone.blocks_spectral_gating.{i}"
        expected |= {f"{prefix}.mlp.fc1", f"{prefix}.mlp.fc2"}
    for i in range(N_ATTENTION_BLOCKS):
        prefix = f"backbone.backbone.blocks_attention.{i}"
        expected |= {
            f"{prefix}.mlp.fc1",
            f"{prefix}.mlp.fc2",
            f"{prefix}.attn.qkv",
            f"{prefix}.attn.proj",
        }

    assert adapted_modules(model) == expected


def test_patch_embedding_and_head_are_never_adapted():
    adapted = adapted_modules(apply_peft_lora(make_model(), LoraAdapterConfig()))
    assert not [n for n in adapted if "embedding" in n], "tokeniser must not be adapted"
    assert not [n for n in adapted if n.startswith(HEAD_PREFIX)], "head must not be adapted"


def test_to_dynamic_projection_is_never_adapted():
    adapted = adapted_modules(apply_peft_lora(make_model(), LoraAdapterConfig()))
    assert not [n for n in adapted if "to_dynamic_projection" in n]


def test_default_target_modules_match_the_backbone():
    """Regression guard: the old split-QKV names match nothing in this backbone."""
    defaults = LoraAdapterConfig().target_modules
    assert defaults == ["fc1", "fc2", "attn.qkv", "attn.proj"]

    module_names = [name for name, _ in make_model().named_modules()]
    for entry in defaults:
        assert any(
            name == entry or name.endswith("." + entry) for name in module_names
        ), f"target_modules entry {entry!r} matches no module; PEFT would ignore it silently"


def test_bare_proj_would_capture_the_tokeniser():
    """Why the dotted 'attn.proj' form is required rather than a bare 'proj'."""
    cfg = LoraAdapterConfig(target_modules=["fc1", "fc2", "qkv", "proj"])
    adapted = adapted_modules(apply_peft_lora(make_model(), cfg))
    assert "backbone.embedding.patch_embed.proj" in adapted


# ---------------------------------------------------------------------------
# The head stays trainable
# ---------------------------------------------------------------------------


def test_head_is_trainable_and_backbone_is_not():
    model = apply_peft_lora(make_model(), LoraAdapterConfig())

    for name, param in model.named_parameters():
        is_adapter = ".lora_" in name
        # PEFT keeps a frozen original alongside the trainable copy.
        is_trainable_head_copy = "modules_to_save" in name

        if is_adapter or is_trainable_head_copy:
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


def test_every_head_module_has_a_trainable_copy():
    model = make_model()
    expected = set(discover_head_modules(model))
    assert expected == {"head_cls_token", "head_linear", "head_unembed"}

    peft_model = apply_peft_lora(model, LoraAdapterConfig())
    saved = {
        name.replace("base_model.model.", "").split(".modules_to_save")[0]
        for name, _ in peft_model.named_parameters()
        if "modules_to_save" in name
    }
    assert saved == expected


def test_parameter_free_head_modules_are_not_duplicated():
    """head_dropout carries no parameters, so PEFT need not wrap it."""
    model = make_model()
    assert isinstance(model.head_dropout, nn.Dropout) or model.head_dropout is None
    assert "head_dropout" not in discover_head_modules(model)


# ---------------------------------------------------------------------------
# An optimizer step actually moves the right tensors
# ---------------------------------------------------------------------------


def test_optimizer_step_updates_head_and_lora_b_only():
    torch.manual_seed(0)
    model = apply_peft_lora(make_model(), LoraAdapterConfig())

    tracked = {
        name: param
        for name, param in model.named_parameters()
        if "modules_to_save" in name or ".lora_B" in name
    }
    frozen = {
        name: param
        for name, param in model.named_parameters()
        if name.endswith("attn.qkv.base_layer.weight") or name.endswith("mlp.fc1.base_layer.weight")
    }
    assert tracked and frozen

    before = {name: param.detach().clone() for name, param in {**tracked, **frozen}.items()}

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0)
    model(make_batch()).sum().backward()
    optimizer.step()

    # lora_B starts at zero, so it only moves if gradient reaches it through the head.
    for name, param in tracked.items():
        assert not torch.equal(param, before[name]), f"{name} did not change"
    for name, param in frozen.items():
        assert torch.equal(param, before[name]), f"frozen {name} changed"


def test_class_token_receives_gradient():
    """The specific symptom of the original bug: cls_token stuck at zeros."""
    model = apply_peft_lora(make_model(pooling="class_token"), LoraAdapterConfig())
    token = dict(model.named_parameters())[
        "base_model.model.head_cls_token.modules_to_save.default.token"
    ]
    assert torch.count_nonzero(token) == 0, "class_token should start at zeros"

    model(make_batch()).sum().backward()
    assert token.grad is not None and torch.count_nonzero(token.grad) > 0


@pytest.mark.parametrize(
    "pooling", ["class_token", "transformer", "attention", "global_average"]
)
def test_all_poolings_build_and_train_one_step(pooling):
    torch.manual_seed(0)
    model = apply_peft_lora(make_model(pooling=pooling), LoraAdapterConfig())

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    output = model(make_batch())
    assert output.shape == (2,)
    output.sum().backward()
    optimizer.step()

    head_params = [p for n, p in model.named_parameters() if "modules_to_save" in n]
    assert head_params, f"{pooling} produced no trainable head parameters"
    assert all(p.grad is not None for p in head_params)


# ---------------------------------------------------------------------------
# Validation of the head_ convention
# ---------------------------------------------------------------------------


def test_head_module_without_prefix_is_rejected():
    model = make_model()
    model.extra_head = nn.Linear(EMBED_DIM, 1)  # missing the head_ prefix

    with pytest.raises(ValueError, match="extra_head"):
        discover_head_modules(model)


def test_parameter_free_module_without_prefix_is_allowed():
    model = make_model()
    model.some_dropout = nn.Dropout(0.1)  # no parameters -> exempt
    assert "some_dropout" not in discover_head_modules(model)


def test_bare_top_level_parameter_is_rejected():
    model = make_model()
    model.head_raw_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))

    with pytest.raises(ValueError, match="head_raw_token"):
        discover_head_modules(model)


def test_head_name_colliding_with_backbone_is_rejected():
    """PEFT matches modules_to_save with a bare endswith, so suffixes collide."""
    model = make_model()
    model.backbone.custom_linear = nn.Linear(EMBED_DIM, EMBED_DIM)

    # "head_linear" is not a suffix of "backbone.custom_linear", but "linear" is
    # -- reproduce the hazard with a name that really does collide.
    model.backbone.my_head_linear = nn.Linear(EMBED_DIM, EMBED_DIM)

    with pytest.raises(ValueError, match="collides"):
        discover_head_modules(model)


# ---------------------------------------------------------------------------
# ClassToken
# ---------------------------------------------------------------------------


def test_class_token_expands_to_batch_size():
    token = ClassToken(EMBED_DIM)
    assert token(1).shape == (1, 1, EMBED_DIM)
    assert token(5).shape == (5, 1, EMBED_DIM)


def test_class_token_forward_dispatches_to_trainable_copy_under_peft():
    """Calling the module must reach the trainable copy, not the frozen original."""
    model = apply_peft_lora(make_model(pooling="class_token"), LoraAdapterConfig())
    wrapper = model.base_model.model.head_cls_token

    output = wrapper(3)
    assert output.shape == (3, 1, EMBED_DIM)
    assert output.requires_grad, "token read must be differentiable"

    output.sum().backward()
    assert wrapper.modules_to_save["default"].token.grad is not None
    assert wrapper.original_module.token.grad is None


def test_class_token_init_modes():
    torch.manual_seed(0)
    assert torch.count_nonzero(ClassToken(EMBED_DIM, init="zeros").token) == 0
    assert torch.count_nonzero(ClassToken(EMBED_DIM, init="randn").token) > 0
    with pytest.raises(ValueError):
        ClassToken(EMBED_DIM, init="uniform")
