import torch
import torch.nn as nn
from einops import rearrange

"""
A simple linear regression model to be used as a baseline for EVE spectral forecasting.

Differences vs RegressionFlareModel:
- Output is a 1D spectrum vector (length = n_wavelengths), not a scalar.
- Otherwise: identical normalization inversion + spatial mean pooling + flatten (C*T) + linear map.
"""


class RegressionEVEModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, channel_order: list[str], scalers: dict):
        """
        Initializes the RegressionEVEModel.

        Args:
            input_dim (int): Size of the flattened input vector after channel and time dims are collapsed, i.e., C*T.
            output_dim (int): Number of wavelength bins in the EVE spectrum (length of the 1D target array).
            channel_order (list[str]): Order of channels as they appear in the input stack (matches Surya dataset output).
            scalers (dict): Per-channel scalers used to inverse transform normalized data back to physical log space.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.channel_order = channel_order
        self.scalers = scalers

    def forward(self, x: dict) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (dict): Batch dictionary containing:
                - x["ts"]: torch.Tensor of shape (b, c, t, w, h)

        Returns:
            torch.Tensor: Predicted spectrum of shape (b, output_dim)
        """

        # Avoid mutating the caller's tensor
        ts = x["ts"].clone()

        # Get dimensions
        b, c, t, w, h = ts.shape

        # Invert normalization to work in physical logarithmic space
        # (match template behavior exactly: in-place per channel, no gradients)
        with torch.no_grad():
            for channel_index, channel in enumerate(self.channel_order):
                ts[:, channel_index, ...] = self.scalers[channel].inverse_transform(
                    ts[:, channel_index, ...]
                )

        # Collapse spatial dimensions and enforce positivity (EUV irradiance/spectra are non-negative)
        ts = ts.abs().mean(dim=[3, 4])  # (b, c, t)

        # Flatten channel and time for linear regression
        ts = rearrange(ts, "b c t -> b (c t)")  # (b, c*t)

        # Linear map to spectrum
        out = self.linear(ts)  # (b, output_dim)
        return out
