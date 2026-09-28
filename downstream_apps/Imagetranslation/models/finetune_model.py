"""
The EUV2MAG fine-tuning model: Surya's backbone plus a decoder head you own.

The backbone comes from ``workshop_infrastructure`` because nobody should re-type Surya's
18 constructor arguments. Everything after it is here, in the app, to edit.

--------------------------------------------------------------------------------
The input side is what makes this app unusual
--------------------------------------------------------------------------------

Surya was pretrained on 13 channels. This app feeds it 3, so the tokenizer
(``backbone.embedding.patch_embed``) is built for 3 and is **not** the pretrained
tokenizer -- it is the pretrained tokenizer restricted to those three channels' weights,
selected out of the checkpoint by ``load_pretrained_weights(channel_indices=...)``.

Two consequences worth understanding before changing anything here:

1. **The restricted tokenizer must train.** ``LinearEmbedding`` is
   ``patch_embed(x) + pos_embed`` with no normalization in between, and the blocks are
   pre-norm, so LayerNorm only ever sees a branch input, never the residual stream. Taking
   3 channels instead of 13 therefore shrinks the tokenizer's output while ``pos_embed`` --
   a fixed-amplitude Fourier buffer -- stays where it is, and position's share of every
   token grows.

   Measured on real normalized data (``tools/measure_token_scale.py``, 3 samples at
   1024px): token std falls from 2.02-2.27 to 1.23-1.33, a ratio of **0.59**, so
   **position's share of each token grows 1.68x**. No downstream LayerNorm can restore that
   ratio, because normalization acts on the sum. The tokenizer is the only layer that can
   rescale its own output; a LoRA adapter cannot, since it acts after tokenization and
   cannot change a per-channel linear map. Hence
   ``model.trainable_backbone_modules: [embedding.patch_embed]`` in the config.

   (An independence argument predicts sqrt(3/13) = 0.48. The measured 0.59 is milder --
   the channels are correlated, and aia304/193/171 are not average channels -- but the
   direction and the size of the effect are what the design turns on, and both hold.)

2. **An earlier version put a Conv3d(3 -> 13) in front of an unmodified 13-channel
   tokenizer instead.** That keeps every pretrained weight, but inserts a randomly
   initialized layer *before* them, so step 0 is not the pretrained function either -- and
   it adds a layer rather than adapting one. Selecting the channels is the same idea
   carried through to the weights.

--------------------------------------------------------------------------------
The one rule the head must follow
--------------------------------------------------------------------------------

**Every trainable head component is a direct attribute named ``head_*``.** Under LoRA,
PEFT freezes the whole model and then re-enables the adapters plus the modules it was told
to save; ``apply_peft_lora()`` finds those by this prefix. A head layer without it is
frozen at its random initialization and the adapters fit a random readout, with nothing in
the loss curve to say so. ``discover_head_modules()`` checks this at startup and names the
attribute to rename.

--------------------------------------------------------------------------------
Shapes
--------------------------------------------------------------------------------

    batch["ts"]              (B, C_in, T, H, W)    normalized EUV stack
    backbone(batch)          (B, L, D)             L = (img_size / patch_size) ** 2
    head_unembed(tokens)     (B, C_out, H, W)      one frame, C_out channels

With the shipped config: C_in=3, T=1, H=W=1024, L=4096, D=1280, C_out=1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from workshop_infrastructure.models.embedding import LinearDecoder, PerceiverDecoder
from workshop_infrastructure.models.finetune_models import build_surya_backbone


class Euv2MagSuryaModel(nn.Module):
    """Surya backbone + a spatial decoder that unembeds tokens back to an image.

    ``HelioSpectformer2D`` in ``workshop_infrastructure`` does the same thing and is the
    right choice for an app that does not need to change it. This model exists so the
    decoder is in the app, where it can be edited without touching infrastructure -- if
    you replace the linear decoder with something convolutional, this is the file.

    Args:
        model_cfg: The ``model:`` section of the loaded config (a ``ModelConfig``).
        out_chans: Number of channels to predict, i.e. ``len(data.target_channels)``.
        **backbone_overrides: Passed to ``build_surya_backbone`` -- the arguments that live
            outside ``ModelConfig``, such as ``use_latitude_in_learned_flow``.
    """

    def __init__(self, model_cfg, out_chans: int, **backbone_overrides):
        super().__init__()

        # nglo=0: this model unembeds every patch token, so the backbone reserves no extra
        # global token. A CLS token would land in the token sequence the decoder reshapes
        # into an image, and the reshape would be off by one.
        self.backbone = build_surya_backbone(model_cfg, nglo=0, **backbone_overrides)

        # ---- The head. Everything below is yours to change. --------------------
        if model_cfg.ft_unembedding_type == "linear":
            self.head_unembed = LinearDecoder(
                patch_size=model_cfg.patch_size,
                out_chans=out_chans,
                embed_dim=model_cfg.embed_dim,
            )
        elif model_cfg.ft_unembedding_type == "perceiver":
            self.head_unembed = PerceiverDecoder(
                embed_dim=model_cfg.embed_dim,
                patch_size=model_cfg.patch_size,
                out_chans=out_chans,
            )
        else:
            raise ValueError(
                "model.ft_unembedding_type must be 'linear' or 'perceiver', got "
                f"{model_cfg.ft_unembedding_type!r}."
            )
        # ------------------------------------------------------------------------

    def forward(self, batch: dict) -> torch.Tensor:
        """Map a batch dict to a predicted image of shape (B, C_out, H, W)."""
        tokens = self.backbone(batch)  # (B, L, D)
        return self.head_unembed(tokens)  # -> (B, C_out, H, W)
