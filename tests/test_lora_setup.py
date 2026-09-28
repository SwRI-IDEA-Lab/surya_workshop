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

from conftest import (
    DEPTH,
    EMBED_DIM,
    IN_CHANS,
    N_ATTENTION_BLOCKS,
    N_SPECTRAL_BLOCKS,
    PATCH_SIZE,
    make_batch,
    make_model,
)
from workshop_infrastructure.configs import LoraAdapterConfig
from workshop_infrastructure.models.finetune_models import ClassToken
from workshop_infrastructure.utils import (
    HEAD_PREFIX,
    apply_peft_lora,
    discover_head_modules,
    resolve_trainable_backbone_modules,
)


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


# ---------------------------------------------------------------------------
# Backbone layers an app deliberately keeps trainable
# ---------------------------------------------------------------------------
#
# LoRA freezes the backbone, which is the point. But an app that changes the *shape* of
# an input layer -- a tokenizer rebuilt for a subset of the 13 pretraining channels --
# has to train it: the sliced tokenizer is not the function the backbone was trained to
# consume, and a low-rank adapter cannot stand in for it, because it acts after
# tokenization and cannot change a per-channel linear map. Without this list the
# tokenizer is frozen at that slice and nothing in the logs says so.

TOKENIZER = "embedding.patch_embed"


def _tokenizer_params(model):
    return {n: p for n, p in model.named_parameters() if "patch_embed" in n}


def test_tokenizer_is_frozen_by_default():
    """The default must stay "LoRA freezes the backbone" -- this is opt-in only."""
    model = apply_peft_lora(make_model(), LoraAdapterConfig())
    assert not any(p.requires_grad for p in _tokenizer_params(model).values())


def test_named_backbone_module_stays_trainable():
    model = apply_peft_lora(
        make_model(), LoraAdapterConfig(), trainable_backbone_modules=[TOKENIZER]
    )
    trainable = {n for n, p in _tokenizer_params(model).items() if p.requires_grad}
    assert trainable, "tokenizer must be trainable when named"
    # Both the conv weight and its bias, not just one of them.
    assert sum(n.endswith(("proj.weight", "proj.bias")) for n in trainable) == 2


def test_naming_one_backbone_module_does_not_unfreeze_the_rest():
    model = apply_peft_lora(
        make_model(), LoraAdapterConfig(), trainable_backbone_modules=[TOKENIZER]
    )
    leaked = [
        n
        for n, p in model.named_parameters()
        if p.requires_grad
        and "patch_embed" not in n
        and "lora_" not in n
        and HEAD_PREFIX not in n
    ]
    assert leaked == [], f"LoRA must still freeze the rest of the backbone, got {leaked}"


def test_gradient_reaches_the_trainable_tokenizer():
    """requires_grad is necessary but not sufficient -- PEFT dispatches through a
    wrapper, so the *trainable copy* is what must receive the gradient."""
    model = apply_peft_lora(
        make_model(pooling="global_average"),
        LoraAdapterConfig(),
        trainable_backbone_modules=[TOKENIZER],
    )
    model(make_batch()).sum().backward()

    wrapper = model.base_model.model.backbone.embedding.patch_embed
    assert wrapper.modules_to_save["default"].proj.weight.grad is not None
    assert wrapper.original_module.proj.weight.grad is None


def test_trainable_tokenizer_gets_no_lora_adapters():
    """It should be fully trainable *instead of* adapted, never both."""
    model = apply_peft_lora(
        make_model(), LoraAdapterConfig(), trainable_backbone_modules=[TOKENIZER]
    )
    assert [n for n in adapted_modules(model) if "patch_embed" in n] == []


def test_trainable_tokenizer_adds_the_expected_parameter_count():
    base = apply_peft_lora(make_model(), LoraAdapterConfig())
    with_tok = apply_peft_lora(
        make_model(), LoraAdapterConfig(), trainable_backbone_modules=[TOKENIZER]
    )
    count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    conv = EMBED_DIM * IN_CHANS * 1 * PATCH_SIZE * PATCH_SIZE + EMBED_DIM  # time_dim=1
    assert count(with_tok) - count(base) == conv


def test_unknown_backbone_module_name_raises_instead_of_doing_nothing():
    with pytest.raises(ValueError, match="matches no module"):
        resolve_trainable_backbone_modules(make_model(), ["patch_embedd"])


def test_ambiguous_backbone_module_name_raises():
    """PEFT matches modules_to_save by suffix, so an ambiguous name wraps too much."""
    with pytest.raises(ValueError, match="matches 3 modules"):
        resolve_trainable_backbone_modules(make_model(), ["norm1"])


def test_empty_list_resolves_to_empty():
    assert resolve_trainable_backbone_modules(make_model(), []) == []


def test_trainable_tokenizer_is_not_in_the_head_optimizer_group():
    """It starts from pretrained weights, so it belongs at the backbone learning rate,
    not the 10x rate a randomly-initialised read-out needs."""
    from downstream_apps.template.lightning_modules.pl_finetune import (
        FlareFinetuneLightningModule as M,
    )

    model = apply_peft_lora(
        make_model(), LoraAdapterConfig(), trainable_backbone_modules=[TOKENIZER]
    )
    tok = next(n for n, p in _tokenizer_params(model).items() if p.requires_grad)
    assert not M._is_head_parameter(tok)
