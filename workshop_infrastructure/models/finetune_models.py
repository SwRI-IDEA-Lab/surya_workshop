import torch
from torch import nn

from workshop_infrastructure.models.helio_spectformer import HelioSpectFormer
from workshop_infrastructure.models.embedding import LinearDecoder, PerceiverDecoder


_VALID_POOLINGS = {"global_average", "global_max", "attention", "transformer", "class_token"}


class HelioSpectformer1D(nn.Module):
    """
    Fine-tuning wrapper for 1D outputs (e.g. regression or classification).

    Holds a frozen-or-trainable HelioSpectFormer backbone and adds a pooling
    layer plus a linear head on top. Only the head-specific parameters are
    defined here; all backbone parameters are forwarded to HelioSpectFormer.
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
        nglo: int = 0,
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
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        self.penultimate_linear_layer_enabled = penultimate_linear_layer

        if pooling == "attention":
            self.attn_pool = nn.MultiheadAttention(embed_dim, num_penultimate_heads, dropout=dropout)

        elif pooling == "transformer":
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_penultimate_heads,
                dim_feedforward=embed_dim,
                dropout=dropout,
            )
            self.downstream_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_penultimate_transformer_layers
            )

        elif pooling == "class_token":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if penultimate_linear_layer:
            self.linear = nn.Linear(embed_dim, embed_dim)

        self.unembed = nn.Linear(embed_dim, num_outputs)

    def forward(self, batch):
        if self.pooling == "class_token":
            tokens = self.backbone.forward_with_cls_token(batch, self.cls_token)
        else:
            tokens = self.backbone.forward(batch)

        if self.dropout_layer is not None:
            tokens = self.dropout_layer(tokens)

        if self.penultimate_linear_layer_enabled:
            tokens = self.linear(tokens)

        if self.pooling == "global_average":
            agg_tokens = torch.mean(tokens, dim=1)
        elif self.pooling == "global_max":
            agg_tokens, _ = torch.max(tokens, dim=1)
        elif self.pooling == "attention":
            tokens = tokens.permute(1, 0, 2)
            tokens, _ = self.attn_pool(query=tokens, key=tokens, value=tokens)
            agg_tokens = tokens.sum(dim=0)
        elif self.pooling == "transformer":
            B = tokens.size(0)
            tokens = torch.cat((self.cls_token.expand(B, -1, -1), tokens), dim=1)
            tokens = self.downstream_transformer(tokens.permute(1, 0, 2))
            agg_tokens = tokens[0, :, :]
        elif self.pooling == "class_token":
            agg_tokens = tokens.squeeze(dim=1)

        if self.dropout_layer is not None:
            agg_tokens = self.dropout_layer(agg_tokens)

        return self.unembed(agg_tokens).squeeze(dim=1)


class HelioSpectformer2D(nn.Module):
    """
    Fine-tuning wrapper for 2D outputs (e.g. image reconstruction or forecasting).

    Holds a HelioSpectFormer backbone and adds a spatial decoder head on top.
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
            self.unembed = LinearDecoder(
                patch_size=patch_size,
                out_chans=ft_out_chans,
                embed_dim=embed_dim,
            )
        elif ft_unembedding_type == "perceiver":
            self.unembed = PerceiverDecoder(
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
        return self.unembed(tokens)  # (B, L, D) -> (B, C, H, W)
