"""
The non-Surya baseline for EUV2MAG.

A per-pixel linear map from the EUV channels to the magnetogram: a 1x1 convolution, which
is exactly a learned linear combination of the input channels at each pixel independently.

It is worth running before the fine-tune, because it answers a question the fine-tune
cannot: how much of the magnetogram is predictable from EUV intensity *at the same pixel*,
with no spatial context at all. Whatever the Surya fine-tune beats this by is what the
spatial and spectral structure in the backbone is buying. A fine-tune that does not beat it
is not working, however plausible its loss curve looks.

Note there is no ``head_`` prefix convention here: this model is never wrapped by PEFT
(there is no pretrained backbone to freeze), so every parameter trains by default.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class Conv2DImageTranslationModel(nn.Module):
    """A 1x1 convolution over the channel-and-time axes.

    Args:
        input_channels: Ordered list of input channel names. Only its length is used; it
            is taken as a list so a misconfigured call fails here rather than in the loss.
        target_channels: Ordered list of target channel names.
        n_input_timestamps: Number of input frames, i.e. ``model.time_embedding.time_dim``.
            Time is folded into the convolution's input channels, so the baseline can use
            several frames even though it has no notion of their order.
    """

    def __init__(
        self,
        input_channels: list[str],
        target_channels: list[str],
        n_input_timestamps: int,
    ):
        super().__init__()
        self.input_channels = list(input_channels)
        self.target_channels = list(target_channels)
        self.n_input_timestamps = n_input_timestamps

        self.conv = nn.Conv2d(
            len(self.input_channels) * n_input_timestamps,
            len(self.target_channels),
            kernel_size=1,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """Map a batch dict to a predicted image of shape (B, C_out, H, W).

        Takes the batch dict rather than a bare tensor so it is interchangeable with
        ``Euv2MagSuryaModel`` and can share the same Lightning module -- an earlier version
        took a tensor and needed a subclass that overrode every step to unwrap the dict.
        """
        x = rearrange(batch["ts"], "b c t h w -> b (c t) h w")
        return self.conv(x)
