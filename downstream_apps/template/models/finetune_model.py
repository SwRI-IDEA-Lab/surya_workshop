"""
The fine-tuning model for the flare template — Surya's backbone plus a head you own.

``models/simple_baseline.py`` is a model you can read top to bottom and change freely.
This file is its counterpart for the fine-tuning path: the backbone comes from
``workshop_infrastructure`` because nobody should re-type Surya's 18 constructor
arguments, but **everything after the backbone is here**, in the app, for you to edit.

If you only want a different head, this is the one file to change.

--------------------------------------------------------------------------------
Two rules the head must follow
--------------------------------------------------------------------------------

1. **Every trainable head component is a direct attribute named ``head_*``.**
   Under LoRA, PEFT freezes the entire model and then re-enables the adapters plus the
   modules it was told to save. ``apply_peft_lora()`` finds those by this prefix. A
   head layer without it is frozen at its random initialization, the adapters fit a
   random readout, and nothing about the loss curve says so.
   ``discover_head_modules()`` checks this at startup and names the attribute to rename.

2. **A bare ``nn.Parameter`` cannot be a head component.** PEFT saves modules by name,
   not parameters. Wrap it in a small module — ``ClassToken`` in
   ``workshop_infrastructure/models/finetune_models.py`` is the worked example — and
   read it by *calling* the module, never through the attribute.

--------------------------------------------------------------------------------
Shapes
--------------------------------------------------------------------------------

    batch["ts"]            (B, C, T, H, W)   normalized SDO stack
    backbone(batch)        (B, L, D)         L = (img_size / patch_size) ** 2 tokens
    pooled                 (B, D)            one vector per sample
    head_unembed(pooled)   (B, num_outputs)  -> squeezed to (B,) when num_outputs == 1

With the shipped config: C=13, T=1, H=W=1024, L=4096, D=1280.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from workshop_infrastructure.models.finetune_models import build_surya_backbone

# The poolings this model implements, in pool() below. The other two that
# HelioSpectformer1D supports ("attention", "transformer") need extra head_* modules, and
# "class_token" needs the backbone to reserve a global token (nglo=1) and to be called
# through backbone.forward_with_cls_token(). Copy any of them from
# workshop_infrastructure/models/finetune_models.py if you want them here.
SUPPORTED_POOLINGS = ("global_average", "global_max")


class FlareSuryaModel(nn.Module):
    """Surya backbone + pooling over patch tokens + a linear read-out.

    Args:
        model_cfg: The ``model:`` section of the loaded config (a ``ModelConfig``).
        num_outputs: Width of the final layer. 1 for scalar regression; set it to the
            number of classes for classification, or to the length of a spectrum.
        **backbone_overrides: Passed through to ``build_surya_backbone`` — the arguments
            that live outside ``ModelConfig``, such as ``use_latitude_in_learned_flow``.

    Raises:
        ValueError: If ``model_cfg.pooling`` is one this model does not implement. It is
            better to say so than to read the config key and quietly ignore it.
    """

    def __init__(self, model_cfg, num_outputs: int = 1, **backbone_overrides):
        super().__init__()

        if model_cfg.pooling not in SUPPORTED_POOLINGS:
            raise ValueError(
                f"model.pooling is {model_cfg.pooling!r}, which FlareSuryaModel does not "
                f"implement. This model supports {list(SUPPORTED_POOLINGS)}.\n"
                "Either add it to pool() in downstream_apps/template/models/"
                "finetune_model.py, or use HelioSpectformer1D from "
                "workshop_infrastructure/models/finetune_models.py, which implements all "
                "five."
            )
        self.pooling = model_cfg.pooling

        # nglo=0: this model pools over the patch tokens rather than reading a CLS token
        # out of the backbone, so the backbone reserves no extra global token.
        self.backbone = build_surya_backbone(model_cfg, nglo=0, **backbone_overrides)

        # ---- The head. Everything below is yours to change. --------------------
        self.head_linear = (
            nn.Linear(model_cfg.embed_dim, model_cfg.embed_dim)
            if model_cfg.penultimate_linear_layer
            else None
        )
        self.head_dropout = nn.Dropout(model_cfg.dropout) if model_cfg.dropout > 0 else None
        self.head_unembed = nn.Linear(model_cfg.embed_dim, num_outputs)
        # ------------------------------------------------------------------------

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """Collapse the token sequence (B, L, D) into one vector per sample (B, D).

        ``global_average`` is the default: every patch of the disk contributes equally, it
        needs no extra parameters, and every token gets gradient from step one — whereas a
        CLS token has to learn what to attend to before it says anything useful, which a
        short fine-tune on a few hundred samples may never get to. ``global_max`` is
        sharper, and flares are local, so it is worth a try.

        This is the method to edit. If what you add needs parameters, give it a ``head_*``
        attribute in ``__init__`` (rule 1 above) and add its name to
        ``SUPPORTED_POOLINGS``.
        """
        if self.pooling == "global_average":
            return tokens.mean(dim=1)
        return tokens.amax(dim=1)  # global_max

    def forward(self, batch: dict) -> torch.Tensor:
        """Map a batch dict to predictions of shape (B,) for ``num_outputs == 1``."""
        tokens = self.backbone(batch)  # (B, L, D)

        if self.pooling == "global_average":
            # mean() and an affine layer commute, so pooling first is the same function
            # of the input, computed on one token instead of L of them. At 64x64 tokens
            # that is 4096x less work in the head and one fewer (B, L, D) activation held
            # for the backward pass.
            pooled = self.pool(tokens)
            if self.head_linear is not None:
                pooled = self.head_linear(pooled)
        else:
            # Every other pooling is non-linear, so the layer has to come first.
            if self.head_linear is not None:
                tokens = self.head_linear(tokens)
            pooled = self.pool(tokens)

        if self.head_dropout is not None:
            pooled = self.head_dropout(pooled)

        return self.head_unembed(pooled).squeeze(dim=1)
