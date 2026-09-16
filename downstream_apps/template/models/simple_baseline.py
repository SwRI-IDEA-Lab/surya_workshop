"""
A simple linear regression model to be used as a baseline for flare forecasting.
"""

import torch
import torch.nn as nn
from einops import rearrange


def destandardize_channels(batch: dict, channel_order: list, scalers: dict) -> dict:
    """Return a new batch dict with 'ts' moved from normalized space to signum-log space.

    This undoes the per-channel z-score ONLY. The signum-log compression applied by the
    dataset is deliberately left in place, so the result is
    ``sign(x*s) * log1p(|x*s|)`` — not raw DN/Gauss. Values spanning many orders of
    magnitude make poor features for a single linear layer, so log space is what the
    baseline wants.

    If you need true physical units (plotting, a physical-space loss), use
    ``HelioNetCDFDataset.inverse_transform_data()`` instead, which undoes both stages.
    See the "THE THREE SPACES" block in ``workshop_infrastructure/datasets/helio.py``.

    Args:
        batch: Batch dict containing at minimum a 'ts' key with shape (B, C, T, H, W).
        channel_order: Channel names in the same order as the C dimension of 'ts'.
        scalers: Dict mapping channel name -> scaler with an inverse_transform method.

    Returns:
        A new batch dict with 'ts' replaced by the de-standardized (signum-log) tensor.
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
            This model expects 'ts' in the batch dict to already be in **signum-log** space
            (channel z-scores undone, log compression retained). Use
            destandardize_channels() to pre-process normalized SDO inputs before passing
            them here (e.g., via the preprocess_fn argument of FlareLightningModule).
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: dict) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (dict): Batch dict with 'ts' of shape (B, C, T, H, W) in signum-log space.

        B - Batch size
        C - Channels
        T - Time steps
        H - Height
        W - Width
        """
        x = x["ts"]

        # Collapse input stack spatially and take absolute value for strictly positive flare fluxes
        x = x.abs().mean(dim=[3, 4])

        # Rearrange in preparation for linear layer
        x = rearrange(x, "b c t -> b (c t)")

        return self.linear(x)