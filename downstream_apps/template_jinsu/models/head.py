import torch
import torch.nn as nn
from torchvision.ops import MLP


class MLPHead(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        # norm_layer,
        # activation_layer,
        dropout,
    ):
        super().__init__()

        self.head = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            norm_layer=nn.LayerNorm,
            activation_layer=nn.GELU,
            dropout=dropout,
        )

    def forward(self, x):
        return self.head(x)
