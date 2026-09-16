from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from workshop_infrastructure.configs import ModelConfig
from torch import nn

from workshop_infrastructure.models.helio_spectformer import HelioSpectFormer
from workshop_infrastructure.models.embedding import LinearDecoder, PerceiverDecoder


_VALID_POOLINGS = {"global_average", "global_max", "attention", "transformer", "class_token"}


class ClassToken(nn.Module):
    """A learnable CLS token, wrapped in a Module so PEFT can keep it trainable.

    PEFT's ``modules_to_save`` accepts module *names* only, so a bare
    ``nn.Parameter`` on the top-level model would stay frozen in the LoRA
    regime.  Wrapping the token in a module lets the ``head_`` convention in
    ``apply_peft_lora()`` cover it like any other head component.

    Always read the token by **calling** the module (``self.head_cls_token(B)``).
    Under PEFT the module is replaced by a wrapper that dispatches ``forward``
    to the trainable copy; reaching for the ``.token`` attribute instead
    delegates in version-dependent ways and can silently hand back the frozen
    original.
    """

    def __init__(self, embed_dim: int, init: str = "zeros"):
        super().__init__()
        if init == "zeros":
            data = torch.zeros(1, 1, embed_dim)
        elif init == "randn":
            data = torch.randn(1, 1, embed_dim)
        else:
            raise ValueError(f"init must be 'zeros' or 'randn', got {init!r}")
        self.token = nn.Parameter(data)

    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """Return the token expanded to ``batch_size``, shape (batch_size, 1, embed_dim).

        The argument is required: PEFT's wrapper forwards at least one
        positional argument, so a zero-argument ``forward()`` would raise.
        """
        return self.token.expand(batch_size, -1, -1)


class HelioSpectformer1D(nn.Module):
    """
    Fine-tuning wrapper for 1D outputs (e.g. regression or classification).

    Holds a frozen-or-trainable HelioSpectFormer backbone and adds a pooling
    layer plus a linear head on top. Only the head-specific parameters are
    defined here; all backbone parameters are forwarded to HelioSpectFormer.

    Every trainable head component is a direct child whose name starts with
    ``head_``.  ``apply_peft_lora()`` discovers them by that prefix and keeps
    them trainable via ``modules_to_save``; see its docstring before adding a
    new head layer.
    """

    def __init__(
        self,
        # --- Backbone ---
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        time_embedding: dict,
        depth: int,
        n_spectral_blocks: int,
        num_heads: int,
        mlp_ratio: float,
        drop_rate: float,
        window_size: int,
        dp_rank: int,
        learned_flow: bool = False,
        use_latitude_in_learned_flow: bool = False,
        init_weights: bool = False,
        checkpoint_layers: list[int] | None = None,
        rpe: bool = False,
        ensemble: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
        # --- Fine-tuning head ---
        dropout: float = 0.1,
        num_outputs: int = 1,
        num_penultimate_transformer_layers: int = 1,
        num_penultimate_heads: int = 8,
        pooling: str = "class_token",
        penultimate_linear_layer: bool = True,
    ):
        super().__init__()

        if pooling not in _VALID_POOLINGS:
            raise ValueError(f"pooling must be one of {_VALID_POOLINGS}, got {pooling!r}")

        # Only "class_token" pooling prepends a global token to the backbone input (via
        # forward_with_cls_token, below); every other pooling must run with nglo=0 or the
        # long-short attention reshape fails. "transformer" pooling also uses a class token,
        # but concatenates it after the backbone, so it is not "class_token" here.
        nglo = 1 if pooling == "class_token" else 0

        self.backbone = HelioSpectFormer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            time_embedding=time_embedding,
            depth=depth,
            n_spectral_blocks=n_spectral_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            window_size=window_size,
            dp_rank=dp_rank,
            learned_flow=learned_flow,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            init_weights=init_weights,
            checkpoint_layers=checkpoint_layers,
            rpe=rpe,
            ensemble=ensemble,
            finetune=True,  # always strip the pretraining decoder
            dtype=dtype,
            nglo=nglo,
        )

        self.pooling = pooling
        self.embed_dim = embed_dim
        self.head_dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.penultimate_linear_layer_enabled = penultimate_linear_layer

        if pooling == "attention":
            self.head_attn_pool = nn.MultiheadAttention(
                embed_dim, num_penultimate_heads, dropout=dropout
            )

        elif pooling == "transformer":
            self.head_cls_token = ClassToken(embed_dim, init="randn")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_penultimate_heads,
                dim_feedforward=embed_dim,
                dropout=dropout,
            )
            self.head_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_penultimate_transformer_layers
            )

        elif pooling == "class_token":
            self.head_cls_token = ClassToken(embed_dim, init="zeros")

        if penultimate_linear_layer:
            self.head_linear = nn.Linear(embed_dim, embed_dim)

        self.head_unembed = nn.Linear(embed_dim, num_outputs)

    def forward(self, batch):
        if self.pooling == "class_token":
            # (1, 1, D) -- forward_with_cls_token expands it over the batch itself.
            tokens = self.backbone.forward_with_cls_token(batch, self.head_cls_token(1))
        else:
            tokens = self.backbone.forward(batch)

        if self.penultimate_linear_layer_enabled:
            tokens = self.head_linear(tokens)

        if self.pooling == "global_average":
            agg_tokens = torch.mean(tokens, dim=1)
        elif self.pooling == "global_max":
            agg_tokens, _ = torch.max(tokens, dim=1)
        elif self.pooling == "attention":
            tokens = tokens.permute(1, 0, 2)
            # Positional, not keyword: under PEFT this module is replaced by a
            # wrapper whose forward requires at least one positional argument.
            tokens, _ = self.head_attn_pool(tokens, tokens, tokens)
            agg_tokens = tokens.sum(dim=0)
        elif self.pooling == "transformer":
            B = tokens.size(0)
            tokens = torch.cat((self.head_cls_token(B), tokens), dim=1)
            tokens = self.head_transformer(tokens.permute(1, 0, 2))
            agg_tokens = tokens[0, :, :]
        elif self.pooling == "class_token":
            agg_tokens = tokens.squeeze(dim=1)

        if self.head_dropout is not None:
            agg_tokens = self.head_dropout(agg_tokens)

        return self.head_unembed(agg_tokens).squeeze(dim=1)

    @classmethod
    def from_config(cls, cfg: "ModelConfig", **overrides) -> "HelioSpectformer1D":
        """Construct from a ModelConfig, with optional field overrides.

        Fields that live outside ModelConfig (e.g. ``dtype``,
        ``use_latitude_in_learned_flow``) should be supplied via ``overrides``.
        """
        kwargs = dict(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_channels,
            embed_dim=cfg.embed_dim,
            time_embedding=dataclasses.asdict(cfg.time_embedding),
            depth=cfg.depth,
            n_spectral_blocks=cfg.spectral_blocks,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            drop_rate=cfg.drop_rate,
            window_size=cfg.window_size,
            dp_rank=cfg.dp_rank,
            learned_flow=cfg.learned_flow,
            init_weights=cfg.init_weights,
            checkpoint_layers=cfg.checkpoint_layers,
            rpe=cfg.rpe,
            ensemble=cfg.ensemble,
            dropout=cfg.dropout,
            pooling=cfg.pooling,
            penultimate_linear_layer=cfg.penultimate_linear_layer,
        )
        kwargs.update(overrides)
        return cls(**kwargs)


class HelioSpectformer2D(nn.Module):
    """
    Fine-tuning wrapper for 2D outputs (e.g. image reconstruction or forecasting).

    Holds a HelioSpectFormer backbone and adds a spatial decoder head on top.

    As with HelioSpectformer1D, every trainable head component is a direct
    child whose name starts with ``head_`` -- see ``apply_peft_lora()``.
    """

    def __init__(
        self,
        # --- Backbone ---
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        time_embedding: dict,
        depth: int,
        n_spectral_blocks: int,
        num_heads: int,
        mlp_ratio: float,
        drop_rate: float,
        window_size: int,
        dp_rank: int,
        learned_flow: bool = False,
        use_latitude_in_learned_flow: bool = False,
        init_weights: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        checkpoint_layers: list[int] | None = None,
        rpe: bool = False,
        # --- Fine-tuning head ---
        ft_unembedding_type: str = "linear",
        ft_out_chans: int = 1,
    ):
        super().__init__()

        self.backbone = HelioSpectFormer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            time_embedding=time_embedding,
            depth=depth,
            n_spectral_blocks=n_spectral_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            window_size=window_size,
            dp_rank=dp_rank,
            learned_flow=learned_flow,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            init_weights=init_weights,
            dtype=dtype,
            checkpoint_layers=checkpoint_layers,
            rpe=rpe,
            finetune=True,  # always strip the pretraining decoder
        )

        if ft_unembedding_type == "linear":
            self.head_unembed = LinearDecoder(
                patch_size=patch_size,
                out_chans=ft_out_chans,
                embed_dim=embed_dim,
            )
        elif ft_unembedding_type == "perceiver":
            self.head_unembed = PerceiverDecoder(
                embed_dim=embed_dim,
                patch_size=patch_size,
                out_chans=ft_out_chans,
            )
        else:
            raise ValueError(
                f"ft_unembedding_type must be 'linear' or 'perceiver', got {ft_unembedding_type!r}"
            )

    def forward(self, batch):
        tokens = self.backbone.forward(batch)
        return self.head_unembed(tokens)  # (B, L, D) -> (B, C, H, W)

    @classmethod
    def from_config(cls, cfg: "ModelConfig", **overrides) -> "HelioSpectformer2D":
        """Construct from a ModelConfig, with optional field overrides.

        Fields that live outside ModelConfig (e.g. ``dtype``,
        ``use_latitude_in_learned_flow``, ``ft_unembedding_type``,
        ``ft_out_chans``) should be supplied via ``overrides``.
        """
        kwargs = dict(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_channels,
            embed_dim=cfg.embed_dim,
            time_embedding=dataclasses.asdict(cfg.time_embedding),
            depth=cfg.depth,
            n_spectral_blocks=cfg.spectral_blocks,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            drop_rate=cfg.drop_rate,
            window_size=cfg.window_size,
            dp_rank=cfg.dp_rank,
            learned_flow=cfg.learned_flow,
            init_weights=cfg.init_weights,
            checkpoint_layers=cfg.checkpoint_layers,
            rpe=cfg.rpe,
        )
        kwargs.update(overrides)
        return cls(**kwargs)
