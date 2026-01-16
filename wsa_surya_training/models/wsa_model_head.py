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
import math


# class WSADecoderHead(nn.Module):
#     """
#     Simple decoder head for converting Surya encoder features to WSA maps.
    
#     This is designed to be lightweight and work on top of a pre-trained
#     Surya encoder. It takes spatial feature maps from the encoder and
#     produces a 2D WSA map output.
    
#     Parameters
#     ----------
#     in_channels : int
#         Number of channels in the encoder output features
#         (depends on the Surya model architecture)
    
#     out_channels : int, default=1
#         Number of output channels (1 for single WSA map)
    
#     use_batch_norm : bool, default=False
#         Whether to apply batch normalization in the decoder
#     """
    
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int = 1,
#         use_batch_norm: bool = False,
#     ):
#         """Initialize the WSA decoder head."""
#         super().__init__()
        
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.use_batch_norm = use_batch_norm
        
#         # Simple 1x1 convolution to map to output channels
#         self.conv_1x1 = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=1,
#             padding=0,
#             bias=True
#         )
        
#         # Optional batch normalization
#         if self.use_batch_norm:
#             self.bn = nn.BatchNorm2d(out_channels)
#         else:
#             self.bn = None
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the decoder.
        
#         Parameters
#         ----------
#         x : torch.Tensor
#             Input features from Surya encoder, shape [B, C_in, H, W]
        
#         Returns
#         -------
#         torch.Tensor
#             WSA map prediction, shape [B, 1, H, W]
#         """
#         # Apply 1x1 convolution
#         out = self.conv_1x1(x)
        
#         # Optional batch normalization
#         if self.bn is not None:
#             out = self.bn(out)
        
#         return out


class WSADecoderHead(nn.Module):
    def __init__(
        self,
        embed_dim=1280,   # Encoder output dimension
        out_chans=1,
        img_size=4096,    # Target size
        patch_size=16,    # Original patch size (used to calc encoder output)
    ):
        super().__init__()
        
        # 1. Project Encoder dims to Decoder dims (e.g., 1280 -> 512)
        # We start with a base filter size like 512 or 256
        base_dim = 512
        self.conv_1x1 = nn.Conv2d(embed_dim, base_dim, kernel_size=1)

        # 2. Upsampling Blocks
        # We need to go from 256x256 -> 4096x4096 (16x scaling)
        # This requires 4 Transposed Convs with stride 2
        self.decoder_blocks = nn.ModuleList([
            # Block 1: 256 -> 512
            nn.Sequential(
                nn.ConvTranspose2d(base_dim, base_dim // 2, kernel_size=2, stride=2),
                nn.BatchNorm2d(base_dim // 2),
                nn.ReLU(inplace=True)
            ),
            # Block 2: 512 -> 1024
            nn.Sequential(
                nn.ConvTranspose2d(base_dim // 2, base_dim // 4, kernel_size=2, stride=2),
                nn.BatchNorm2d(base_dim // 4),
                nn.ReLU(inplace=True)
            ),
            # Block 3: 1024 -> 2048
            nn.Sequential(
                nn.ConvTranspose2d(base_dim // 4, base_dim // 8, kernel_size=2, stride=2),
                nn.BatchNorm2d(base_dim // 8),
                nn.ReLU(inplace=True)
            ),
            # Block 4: 2048 -> 4096
            nn.Sequential(
                nn.ConvTranspose2d(base_dim // 8, base_dim // 16, kernel_size=2, stride=2),
                nn.BatchNorm2d(base_dim // 16),
                nn.ReLU(inplace=True)
            ),
        ])

        # 3. Final Prediction Head
        # Output is now 4096 x 4096 with (base_dim // 16) channels
        self.final_conv = nn.Conv2d(base_dim // 16, out_chans, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Args:
            x: Input feature map [B, 1280, 256, 256]
        """
        # [B, 1280, 256, 256] -> [B, 512, 256, 256]
        x = self.conv_1x1(x)

        # Apply 4 Upsampling Blocks
        # 256 -> 512 -> 1024 -> 2048 -> 4096
        for block in self.decoder_blocks:
            x = block(x)

        # Final projection to 1 channel
        x = self.final_conv(x)
        
        return x

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
            # in_channels=encoder_out_channels,
            out_chans=decoder_out_channels,
            # use_batch_norm=use_batch_norm_decoder,
        )
    
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Forward pass of the complete model.
        
    #     Parameters
    #     ----------
    #     x : torch.Tensor
    #         Input AIA image, shape [B, 1, H, W]
        
    #     Returns
    #     -------
    #     torch.Tensor
    #         WSA map prediction, shape [B, 1, H, W]
    #     """
    #     # Pass through encoder
    #     encoder_out = self.encoder(x)
        
    #     # Pass through decoder
    #     wsa_pred = self.decoder(encoder_out)
        
    #     return wsa_pred

    def forward(self, x):
        """
        Args:
            x: Input Dict or Tensor
        """
        # --- STEP 1: ENCODING ---
        # The WSAModel owns the encoder, so it calls it here.
        encoder_out = self.encoder(x)

        # --- STEP 2: RESHAPING (The fix from before) ---
        # Handle Time Dimension: [B, T, L, D] -> [B, L, D]
        if encoder_out.ndim == 4:
            encoder_out = encoder_out[:, -1, :, :]

        # Handle Spatial Grid: [B, L, D] -> [B, D, H, W]
        # Example: [1, 65536, 1280] -> [1, 1280, 256, 256]
        if encoder_out.ndim == 3:
            B, L, D = encoder_out.shape
            side = int(math.sqrt(L)) 
            # Permute to [B, D, L] then reshape to [B, D, H, W]
            encoder_out = encoder_out.permute(0, 2, 1).reshape(B, D, side, side)

        # --- STEP 3: DECODING ---
        # Pass the reshaped grid to the decoder
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