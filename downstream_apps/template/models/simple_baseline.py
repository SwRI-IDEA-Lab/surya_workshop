import torch
import torch.nn as nn
from einops import rearrange

"""
A simple linear regression model to be used as a baseline for flare forecasting.
"""


def inverse_transform_channels(batch: dict, channel_order: list, scalers: dict) -> dict:
    """Return a new batch dict with 'ts' inverse-transformed to physical log space.

    This converts each SDO channel from its normalized representation back to
    the physical (signum-log) domain before feature extraction. Call this before
    passing a batch to RegressionFlareModel.

    Args:
        batch: Batch dict containing at minimum a 'ts' key with shape (B, C, T, H, W).
        channel_order: Channel names in the same order as the C dimension of 'ts'.
        scalers: Dict mapping channel name -> scaler with an inverse_transform method.

    Returns:
        A new batch dict with 'ts' replaced by the inverse-transformed tensor.
    """
    x = batch["ts"].clone()
    with torch.no_grad():
        for i, channel in enumerate(channel_order):
            x[:, i, ...] = scalers[channel].inverse_transform(x[:, i, ...])
    return {**batch, "ts": x}


class RegressionFlareModel(nn.Module):
    def __init__(self, input_dim: int):
        """
        Initializes the RegressionFlareModel.

        Args:
            input_dim (int): The size of the input vector after channel and time dimensions are flattened.

        Note:
            This model expects 'ts' in the batch dict to already be in physical (log) space.
            Use inverse_transform_channels() to pre-process normalized SDO inputs before
            passing them here (e.g., via the preprocess_fn argument of FlareLightningModule).
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: dict) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (dict): Batch dict with 'ts' of shape (B, C, T, W, H) in physical space.

        B - Batch size
        C - Channels
        T - Time steps
        W - Width
        H - Height
        """
        x = x["ts"]

        # Collapse input stack spatially and take absolute value for strictly positive flare fluxes
        x = x.abs().mean(dim=[3, 4])

        # Rearrange in preparation for linear layer
        x = rearrange(x, "b c t -> b (c t)")

        return self.linear(x)