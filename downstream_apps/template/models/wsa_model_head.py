"""
wsa_model_head.py

Simple decoder head for WSA (Wang-Sheeley-Arge) map prediction.

This module wraps a pre-trained Surya encoder and adds a lightweight decoder head to:
  - Take normalized AIA 193 image input [B, 1, H, W]
  - Pass through Surya encoder to extract features
  - Decode features to spatial WSA map output [B, 1, H, W]

The decoder is a simple approach using:
  - 1x1 convolution to map encoder features to output channel
  - Optional upsampling if needed to match target resolution

This keeps the pre-trained encoder frozen (or partially frozen) and only
trains the lightweight decoder head on WSA-specific data.
"""

import torch
import torch.nn as nn


class WSADecoderHead(nn.Module):
    """
    Simple decoder head for converting Surya encoder features to WSA maps.
    
    This is designed to be lightweight and work on top of a pre-trained
    Surya encoder. It takes spatial feature maps from the encoder and
    produces a 2D WSA map output.
    
    Parameters
    ----------
    in_channels : int
        Number of channels in the encoder output features
        (depends on the Surya model architecture)
    
    out_channels : int, default=1
        Number of output channels (1 for single WSA map)
    
    use_batch_norm : bool, default=False
        Whether to apply batch normalization in the decoder
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        use_batch_norm: bool = False,
    ):
        """Initialize the WSA decoder head."""
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_batch_norm = use_batch_norm
        
        # Simple 1x1 convolution to map to output channels
        self.conv_1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            bias=True
        )
        
        # Optional batch normalization
        if self.use_batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features from Surya encoder, shape [B, C_in, H, W]
        
        Returns
        -------
        torch.Tensor
            WSA map prediction, shape [B, 1, H, W]
        """
        # Apply 1x1 convolution
        out = self.conv_1x1(x)
        
        # Optional batch normalization
        if self.bn is not None:
            out = self.bn(out)
        
        return out


class WSAModel(nn.Module):
    """
    Complete model for WSA map prediction.
    
    Wraps:
      - A pre-trained Surya encoder (frozen or partially trainable)
      - A lightweight WSA decoder head (trainable)
    
    This architecture allows fine-tuning of WSA prediction on top of
    pre-trained solar image understanding from Surya.
    
    Parameters
    ----------
    encoder : nn.Module
        Pre-trained Surya encoder (e.g., SpectFormer encoder)
    
    encoder_out_channels : int
        Number of output channels from the encoder
    
    decoder_out_channels : int, default=1
        Number of output channels from decoder (1 for WSA map)
    
    freeze_encoder : bool, default=True
        Whether to freeze encoder weights (no gradient updates)
    
    use_batch_norm_decoder : bool, default=False
        Whether to use batch normalization in decoder
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        encoder_out_channels: int,
        decoder_out_channels: int = 1,
        freeze_encoder: bool = True,
        use_batch_norm_decoder: bool = False,
    ):
        """Initialize the WSA model."""
        super().__init__()
        
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        
        # Freeze encoder if requested
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Initialize decoder
        self.decoder = WSADecoderHead(
            in_channels=encoder_out_channels,
            out_channels=decoder_out_channels,
            use_batch_norm=use_batch_norm_decoder,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the complete model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input AIA image, shape [B, 1, H, W]
        
        Returns
        -------
        torch.Tensor
            WSA map prediction, shape [B, 1, H, W]
        """
        # Pass through encoder
        encoder_out = self.encoder(x)
        
        # Pass through decoder
        wsa_pred = self.decoder(encoder_out)
        
        return wsa_pred
    
    def unfreeze_encoder(self, num_layers: int = None):
        """
        Unfreeze encoder weights for fine-tuning.
        
        Parameters
        ----------
        num_layers : int, optional
            Number of final layers to unfreeze. If None, unfreeze all.
        """
        if num_layers is None:
            # Unfreeze all
            for param in self.encoder.parameters():
                param.requires_grad = True
        else:
            # Unfreeze only last N layers (simplified)
            encoder_params = list(self.encoder.parameters())
            for param in encoder_params[-num_layers:]:
                param.requires_grad = True


class SimpleWSAHead(nn.Module):
    """
    Alternative: Simpler version that assumes encoder already outputs spatial features.
    
    If the Surya encoder already produces [B, C, H, W] spatial features,
    this minimal head just applies a channel reduction.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels from encoder
    
    out_channels : int, default=1
        Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape [B, C, H, W]
        
        Returns
        -------
        torch.Tensor
            Output of shape [B, 1, H, W]
        """
        return self.conv(x)