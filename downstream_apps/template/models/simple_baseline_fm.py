import torch
import torch.nn as nn
from einops import rearrange
from typing import Dict
from torch import squeeze


"""
A simple linear regression model to be used as a baseline for flare forecasting.
"""
# needed 2d conv

class RegressionFlilament(nn.Module):
    def __init__(self, input_dim, channel_order):
        """
        Initializes the RegressionFlareModel.

        Args:
            input_dim (int): The size of the input vector after channel and time dimensions are flattened.
            channel_order (list[str]): List of channel names, defining the order in which channels appear in the input data.
                                       This is used to ensure the inverse transform uses the correct scaler for each channel.
            scalers (dict): A dictionary of scalers, one for each channel, used for inverse transforming the data to physical space.
        """
        super().__init__()
        self.channel_reduction = nn.Conv2d(input_dim, 1, kernel_size=1)
        self.channel_order = channel_order

    def forward(self, x:Dict):
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

        
        b, c, t, w, h = x['ts'].shape



        # Rearange in preparation for linear layer
        x['ts'] = rearrange(x['ts'], "b c t w h -> b (c t) w h")

        x['ts'] = self.channel_reduction(x['ts'])
        x['ts'] = squeeze(x['ts'],1)
        print(x['ts'].shape)
       

        return x['ts']