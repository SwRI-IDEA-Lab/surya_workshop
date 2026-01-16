import torch
from torch import nn
from torchvision.ops import MLP
from itertools import chain

from torch.utils.checkpoint import checkpoint

from surya.models.helio_spectformer import HelioSpectFormer

from surya.models.embedding import (
    LinearDecoder,
    PerceiverDecoder,
)


class HelioSpectformerMLPHead(HelioSpectFormer):
    def __init__(
        self,
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
        finetune: bool = False,
        nglo: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        # Put finetuning additions below this line
        global_class_token: bool = True,
        in_channels: int = 1280,
        hidden_channels: list[int] | None = None,
        dropout: float = 0.5,
    ):
        super().__init__(
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
            finetune=finetune,
            dtype=dtype,
            nglo=nglo,
        )
        self.global_class_token = global_class_token
        self.head = MLP(
            in_channels=self.embed_dim,
            hidden_channels=hidden_channels,
            norm_layer=nn.LayerNorm,
            activation_layer=nn.GELU,
            dropout=dropout,
        )

        if self.global_class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def _forward_cls_token(self, batch):
        x = batch["ts"]
        dt = batch["time_delta_input"]
        B, C, T, H, W = x.shape

        if self.learned_flow:
            y_hat_flow = self.learned_flow_model(batch)  # B, C, H, W
            if any(
                [param.requires_grad for param in self.learned_flow_model.parameters()]
            ):
                return y_hat_flow
            else:
                x = torch.concat((x, y_hat_flow.unsqueeze(2)), dim=2)  # B, C, T+1, H, W
                if self.time_embedding["type"] == "perceiver":
                    dt = torch.cat((dt, batch["lead_time_delta"].reshape(-1, 1)), dim=1)

        # embed the data
        tokens = self.embedding(x, dt)

        if self.ensemble:
            raise NotImplementedError(
                "Use of CLS token has not been implemented with ensemble modifications."
            )
        else:
            noise = None

        # pass the time series through the encoder
        for i, blk in enumerate(
            chain(self.backbone.blocks_spectral_gating, self.backbone.blocks_attention)
        ):
            if i == self.backbone.n_spectral_blocks:
                tokens = torch.cat(
                    (
                        self.cls_token.expand(B, 1, self.embed_dim),
                        tokens,
                    ),
                    dim=1,
                )
            if i in self.backbone._checkpoint_layers:
                tokens = checkpoint(blk, tokens, noise, use_reentrant=False)
            else:
                tokens = blk(tokens, noise)
        tokens = tokens[:, [0], :]

        return tokens

    def forward(self, batch):
        if self.global_class_token:
            tokens = self._forward_cls_token(batch)
            out = self.head(tokens)
        else:
            tokens = super().forward(batch=batch)
            token_pooled = tokens.mean(dim=[1, 2])  # B, D
            out = self.head(token_pooled)
        return out
