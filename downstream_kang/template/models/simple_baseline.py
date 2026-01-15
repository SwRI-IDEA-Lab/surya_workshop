# models/simple_baseline.py
import torch
import torch.nn as nn
from torchvision.ops import MLP


class ClsFlareBaseLine(nn.Module):
    def __init__(
        self,
        in_channels: int = 13,
        n_statistics: int = 4,
        hidden_channels: list[int] = None,
        dropout: float = 0.5,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [52, 26, 1],

        feature_dim = in_channels * n_statistics  # 13 * 4 = 52

        # MLP
        self.classifier = MLP(
            in_channels=feature_dim,        # 52
            hidden_channels=hidden_channels,  # [128, 64, 1]
            activation_layer=nn.ReLU,
            norm_layer=nn.BatchNorm1d,
            dropout=dropout,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=[2, 3, 4])      # [B, C]
        x_min = torch.amin(x, dim=[2, 3, 4])  # [B, C]
        x_max = torch.amax(x, dim=[2, 3, 4])  # [B, C]
        x_std = x.std(dim=[2, 3, 4])        # [B, C]

        features = torch.cat([x_mean, x_min, x_max, x_std], dim=1)  # [B, C*4]

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:  # [B, C, T, H, W]
            x = self.extract_features(x)  # [B, C*4]
        elif x.dim() == 2:  # [B, C*4]
            pass
        else:
            raise ValueError(
                f"Expected input with 5 or 2 dimensions, got {x.dim()}")

        return self.classifier(x)
