import torch
import torch.nn as nn

"""
Simple segmentation models to be used as baselines for flare ribbon detection.
"""


class SegmentationRibbonModel(nn.Module):
    def __init__(self, input_dim, channel_order, scalers):
        """
        Initializes the SegmentationRibbonModel.

        A truly simple baseline: just one convolutional layer that learns a weighted combination
        of input channels to predict ribbon presence at each pixel.

        Args:
            input_dim (int): For compatibility, this is ignored. The model automatically determines
                           the number of channels from channel_order.
            channel_order (list[str]): List of channel names, defining the order in which channels appear in the input data.
                                       This is used to ensure the inverse transform uses the correct scaler for each channel.
            scalers (dict): A dictionary of scalers, one for each channel, used for inverse transforming the data to physical space.
        """
        super().__init__()
        self.channel_order = channel_order
        self.scalers = scalers

        # Determine number of input channels from channel_order (ignoring input_dim for segmentation)
        input_channels = len(channel_order)

        # Single 1x1 convolution: learns a weighted combination of input channels
        # This is essentially a per-pixel linear classifier that looks at all channels at each location
        # and predicts: "Is there a ribbon here?"
        # Input: (b, input_channels, h, w) -> Output: (b, 1, h, w)
        self.conv = nn.Conv2d(input_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor or dict): Input tensor of shape (b, c, t, h, w), or a batch dict with key "ts".

        b - Batch size
        c - Channels
        t - Time steps
        h - Height
        w - Width

        Returns:
            torch.Tensor: Segmentation mask of shape (b, 1, h, w) with values in [0, 1].
        """

        # Handle dict input (for compatibility with Lightning module)
        if isinstance(x, dict):
            x = x["ts"]

        # Avoid mutating the caller's tensor
        x = x.clone()

        # Get dimensions
        b, c, t, _, _ = x.shape

        # Invert normalization to work in physical logarithmic space
        with torch.no_grad():
            for channel_index, channel in enumerate(self.channel_order):
                x[:, channel_index, ...] = self.scalers[channel].inverse_transform(
                    x[:, channel_index, ...]
                )

        # Take absolute value for strictly positive values
        x = x.abs()

        # Collapse time dimension by taking the mean across time
        # (b, c, t, h, w) -> (b, c, h, w)
        x = x.mean(dim=2)

        # Apply single convolutional layer + sigmoid
        # This learns: weight_1 * channel_1 + weight_2 * channel_2 + ... + bias
        # Then sigmoid squashes to [0, 1] probability
        x = self.conv(x)
        out = self.sigmoid(x)

        return out


class ThresholdRibbonModel(nn.Module):
    def __init__(self, input_dim, channel_order, scalers, channel_name="1600", sigma=5.0):
        """
        Initializes the ThresholdRibbonModel.

        A non-learned baseline that uses intensity thresholding on a single channel
        to detect ribbons. The threshold is computed using robust statistics (median + sigma * MAD).

        Args:
            input_dim (int): For compatibility, this is ignored (non-learned model has no parameters).
            channel_order (list[str]): List of channel names, defining the order in which channels appear in the input data.
            scalers (dict): A dictionary of scalers, one for each channel, used for inverse transforming the data to physical space.
            channel_name (str): Name of the channel to use for thresholding (e.g., "1600" for AIA 1600Å). Default: "1600".
            sigma (float): Number of standard deviations above the median for threshold. Default: 5.0.
        """
        super().__init__()
        self.channel_order = channel_order
        self.scalers = scalers
        self.channel_name = channel_name
        self.sigma = sigma

        # Find the index of the target channel
        self.channel_idx = None
        for idx, channel in enumerate(channel_order):
            if channel_name in channel:
                self.channel_idx = idx
                break

        if self.channel_idx is None:
            raise ValueError(f"Channel '{channel_name}' not found in channel_order: {channel_order}")

    def forward(self, x):
        """
        Performs a forward pass through the model using intensity thresholding.

        Args:
            x (torch.Tensor): Input tensor of shape (b, c, t, h, w).

        b - Batch size
        c - Channels
        t - Time steps
        h - Height
        w - Width

        Returns:
            torch.Tensor: Binary segmentation mask of shape (b, 1, h, w) with values in [0, 1].
        """

        # Avoid mutating the caller's tensor
        x = x.clone()

        # Get dimensions
        b, c, t, _, _ = x.shape

        # Invert normalization to work in physical logarithmic space
        with torch.no_grad():
            for channel_index, channel in enumerate(self.channel_order):
                x[:, channel_index, ...] = self.scalers[channel].inverse_transform(
                    x[:, channel_index, ...]
                )

        # Take absolute value for strictly positive values
        x = x.abs()

        # Collapse time dimension by taking the mean across time
        # (b, c, t, h, w) -> (b, c, h, w)
        x = x.mean(dim=2)

        # Extract the target channel (e.g., AIA 1600Å)
        target_channel = x[:, self.channel_idx:self.channel_idx+1, :, :]  # (b, 1, h, w)

        # Apply robust thresholding per image in the batch
        masks = []
        for i in range(b):
            img = target_channel[i, 0]  # (h, w)

            # Compute robust statistics: median and MAD (Median Absolute Deviation)
            median = torch.median(img)
            mad = torch.median(torch.abs(img - median))
            std_estimate = 1.4826 * mad  # Convert MAD to std estimate

            # Compute threshold: median + sigma * std
            threshold_value = median + self.sigma * std_estimate

            # Create binary mask
            mask = (img > threshold_value).float()  # (h, w)
            masks.append(mask.unsqueeze(0))  # (1, h, w)

        # Stack all masks
        out = torch.stack(masks, dim=0)  # (b, 1, h, w)

        return out


class SoftThresholdRibbonModel(nn.Module):
    def __init__(self, input_dim, channel_order, scalers, channel_name="1600"):
        """
        Initializes the SoftThresholdRibbonModel.

        A differentiable baseline that learns an adaptive threshold on a single channel.
        Uses a soft (sigmoid-based) threshold instead of hard threshold, making it trainable.

        Args:
            input_dim (int): For compatibility, this is ignored.
            channel_order (list[str]): List of channel names, defining the order in which channels appear in the input data.
            scalers (dict): A dictionary of scalers, one for each channel, used for inverse transforming the data to physical space.
            channel_name (str): Name of the channel to use for thresholding (e.g., "1600" for AIA 1600Å). Default: "1600".
        """
        super().__init__()
        self.channel_order = channel_order
        self.scalers = scalers
        self.channel_name = channel_name

        # Find the index of the target channel
        self.channel_idx = None
        for idx, channel in enumerate(channel_order):
            if channel_name in channel:
                self.channel_idx = idx
                break

        if self.channel_idx is None:
            raise ValueError(f"Channel '{channel_name}' not found in channel_order: {channel_order}")

        # Learnable parameters: threshold and temperature (steepness of sigmoid)
        # Initialize threshold to a reasonable value (will be learned)
        self.threshold = nn.Parameter(torch.tensor(0.0))
        # Temperature controls how "hard" the threshold is (lower = steeper, higher = softer)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """
        Performs a forward pass using a differentiable soft threshold.

        Args:
            x (torch.Tensor): Input tensor of shape (b, c, t, h, w).

        Returns:
            torch.Tensor: Probabilistic segmentation mask of shape (b, 1, h, w) with values in [0, 1].
        """

        # Avoid mutating the caller's tensor
        x = x.clone()

        # Get dimensions
        b, c, t, _, _ = x.shape

        # Invert normalization to work in physical logarithmic space
        with torch.no_grad():
            for channel_index, channel in enumerate(self.channel_order):
                x[:, channel_index, ...] = self.scalers[channel].inverse_transform(
                    x[:, channel_index, ...]
                )

        # Take absolute value for strictly positive values
        x = x.abs()

        # Collapse time dimension by taking the mean across time
        # (b, c, t, h, w) -> (b, c, h, w)
        x = x.mean(dim=2)

        # Extract the target channel (e.g., AIA 1600Å)
        target_channel = x[:, self.channel_idx:self.channel_idx+1, :, :]  # (b, 1, h, w)

        # Apply soft (differentiable) threshold
        # sigmoid((intensity - threshold) / temperature)
        # When temperature is small, this approximates a hard threshold
        # When intensity > threshold, output approaches 1
        # When intensity < threshold, output approaches 0
        out = torch.sigmoid((target_channel - self.threshold) / torch.abs(self.temperature))

        return out