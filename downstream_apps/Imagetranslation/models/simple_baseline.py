import torch
import torch.nn as nn
from einops import rearrange

"""
A simple linear regression model to be used as a baseline for flare forecasting.
"""
class RegressionFlareModel(nn.Module):
    def __init__(self, input_dim, channel_order, scalers):
        """
        Initializes the RegressionFlareModel.

        Args:
            input_dim (int): The size of the input vector after channel and time dimensions are flattened.
            channel_order (list[str]): List of channel names, defining the order in which channels appear in the input data.
                                       This is used to ensure the inverse transform uses the correct scaler for each channel.
            scalers (dict): A dictionary of scalers, one for each channel, used for inverse transforming the data to physical space.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.channel_order = channel_order
        self.scalers = scalers

    def forward(self, x):
        """
        Performs a forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape (b, c, t, w, h).

        b - Batch size
        c - Channels
        t - Time steps
        w - Width
        h - Height
        """

        # Avoid mutating the caller's tensor
        x = x.clone()

        # Get dimensions
        b, c, t, w, h = x.shape

        # Invert normalization to work in physical logarithmic space
        with torch.no_grad():
            for channel_index, channel in enumerate(self.channel_order):
                x[:, channel_index, ...] = self.scalers[channel].inverse_transform(
                    x[:, channel_index, ...]
                )

        # Collapse input stack spatially and take absolute value for strictly positive flare fluxes
        x = x.abs().mean(dim=[3,4])

        # Rearange in preparation for linear layer
        x = rearrange(x, "b c t -> b (c t)")

        out = self.linear(x)
        return out


class Conv2DImageTranslationModel(nn.Module):
    """
    Simple image translation baseline.
    Flattens time into channels and applies a 1x1 conv (per-pixel linear map).
    """

    def __init__(self, input_channels, target_channels, n_input_timestamps):
        """
        Args:
            input_channels (list[str]): Ordered list of input channels.
            target_channels (list[str]): Ordered list of target channels.
            n_input_timestamps (int): Number of input timesteps.
        """
        super().__init__()
        self.input_channels = input_channels
        self.target_channels = target_channels
        self.n_input_timestamps = n_input_timestamps

        in_channels = len(input_channels) * n_input_timestamps
        out_channels = len(target_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Performs a forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape (b, c, t, h, w).

        b - Batch size
        c - Channels
        t - Time steps
        h - Height
        w - Width
        """
        x = x.clone()
        b, c, t, h, w = x.shape
        x = x.reshape(b, c * t, h, w)

        return self.conv(x)

