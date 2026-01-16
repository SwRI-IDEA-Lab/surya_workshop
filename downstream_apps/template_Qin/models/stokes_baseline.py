"""
A simple baseline model for Stokes profile regression.

This model takes synthesized Stokes profiles from Surya magnetograms as input
and predicts Stokes profiles that match those synthesized from observed HMI magnetograms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StokesBaselineModel(nn.Module):
    """
    Simple baseline model for Stokes profile regression.
    
    This model processes Stokes profiles (I, Q, U, V) pixel-wise using a simple
    convolutional or fully-connected architecture.
    
    Input:  [B, 4, n_wavelengths, H, W] - Stokes profiles from Surya
    Output: [B, 4, n_wavelengths, H, W] - Predicted Stokes profiles
    """
    
    def __init__(
        self,
        n_wavelengths: int = 50,
        hidden_dim: int = 128,
        use_conv: bool = True,
    ):
        """
        Initialize the StokesBaselineModel.
        
        Args:
            n_wavelengths: Number of wavelength points in the Stokes profiles
            hidden_dim: Hidden dimension for the model
            use_conv: If True, use convolutional layers; if False, use fully-connected layers
        """
        super().__init__()
        self.n_wavelengths = n_wavelengths
        self.hidden_dim = hidden_dim
        self.use_conv = use_conv
        
        if use_conv:
            # Convolutional approach: process wavelength dimension with 1D convolutions
            # Input: [B, 4, n_wavelengths, H, W]
            # Process each Stokes parameter independently
            
            # 1D convolution along wavelength dimension
            self.conv1 = nn.Conv1d(
                in_channels=4,  # 4 Stokes parameters
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1
            )
            
            self.conv2 = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1
            )
            
            self.conv3 = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=4,  # Output 4 Stokes parameters
                kernel_size=3,
                padding=1
            )
            
            self.activation = nn.ReLU()
        else:
            # Fully-connected approach: process each pixel independently
            # Flatten wavelength dimension: [B, 4, n_wavelengths, H, W] -> [B*H*W, 4*n_wavelengths]
            input_dim = 4 * n_wavelengths
            
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 4 * n_wavelengths)
            
            self.activation = nn.ReLU()
    
    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            batch: Dictionary containing 'stokes_input' key with shape [B, 4, n_wavelengths, H, W]
        
        Returns:
            Predicted Stokes profiles [B, 4, n_wavelengths, H, W]
        """
        x = batch['stokes_input']  # [B, 4, n_wavelengths, H, W]
        B, C, N_w, H, W = x.shape
        
        if self.use_conv:
            # Convolutional approach
            # Reshape: [B, 4, n_wavelengths, H, W] -> [B*H*W, 4, n_wavelengths]
            x = x.permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, 4, n_wavelengths]
            x = x.view(B * H * W, C, N_w)  # [B*H*W, 4, n_wavelengths]
            
            # Apply 1D convolutions along wavelength dimension
            x = self.activation(self.conv1(x))  # [B*H*W, hidden_dim, n_wavelengths]
            x = self.activation(self.conv2(x))  # [B*H*W, hidden_dim, n_wavelengths]
            x = self.conv3(x)  # [B*H*W, 4, n_wavelengths]
            
            # Reshape back: [B*H*W, 4, n_wavelengths] -> [B, 4, n_wavelengths, H, W]
            x = x.view(B, H, W, C, N_w)  # [B, H, W, 4, n_wavelengths]
            x = x.permute(0, 3, 4, 1, 2).contiguous()  # [B, 4, n_wavelengths, H, W]
        else:
            # Fully-connected approach
            # Flatten: [B, 4, n_wavelengths, H, W] -> [B*H*W, 4*n_wavelengths]
            x = x.permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, 4, n_wavelengths]
            x = x.view(B * H * W, C * N_w)  # [B*H*W, 4*n_wavelengths]
            
            # Apply fully-connected layers
            x = self.activation(self.fc1(x))  # [B*H*W, hidden_dim]
            x = self.activation(self.fc2(x))  # [B*H*W, hidden_dim]
            x = self.fc3(x)  # [B*H*W, 4*n_wavelengths]
            
            # Reshape back: [B*H*W, 4*n_wavelengths] -> [B, 4, n_wavelengths, H, W]
            x = x.view(B, H, W, C, N_w)  # [B, H, W, 4, n_wavelengths]
            x = x.permute(0, 3, 4, 1, 2).contiguous()  # [B, 4, n_wavelengths, H, W]
        
        return x
