"""
wsa_metrics.py

Metrics for WSA map regression training.

This module defines loss and evaluation metrics for training WSA (Wang-Sheeley-Arge)
full-disk maps. The task is image-to-image regression where:
  - Input: AIA 193 image [B, H, W] or [B, 1, H, W]
  - Target: WSA map [B, H, W] or [B, 1, H, W]

The WSAMetrics class provides:
  - train_loss: MSE loss for backpropagation
  - train_metrics: MAE for monitoring during training
  - val_metrics: MAE for validation monitoring
"""

import torch
import torchmetrics as tm


class WSAMetrics:
    """
    Metrics for WSA map regression.
    
    Follows the same pattern as template_metrics.py but optimized for 2D map outputs.
    """
    
    def __init__(self, mode: str):
        """
        Initialize WSAMetrics class.
        
        Parameters
        ----------
        mode : str
            Mode to use for metric evaluation. Can be "train_loss",
            "train_metrics", or "val_metrics".
        """
        self.mode = mode
        
        # Cache torchmetrics instances
        self._mae = tm.MeanAbsoluteError()
    
    def _ensure_device(self, preds: torch.Tensor):
        """Move metric module to same device as predictions."""
        if self._mae.device != preds.device:
            self._mae = self._mae.to(preds.device)
    
    def _flatten_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Flatten spatial dimensions for metric computation.
        
        Handles both [B, H, W] and [B, 1, H, W] shapes by flattening to [B*H*W].
        
        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor of shape [B, H, W] or [B, 1, H, W]
        
        Returns
        -------
        torch.Tensor
            Flattened tensor of shape [B*H*W]
        """
        if tensor.ndim == 4:
            # [B, 1, H, W] -> [B, H, W]
            tensor = tensor.squeeze(1)
        
        # [B, H, W] -> [B*H*W]
        return tensor.reshape(-1)
    
    def train_loss(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate loss metrics for training.
        
        Uses MSE loss for backpropagation.
        
        Parameters
        ----------
        preds : torch.Tensor
            Model predictions, shape [B, H, W] or [B, 1, H, W]
        target : torch.Tensor
            Ground truth WSA maps, shape [B, H, W] or [B, 1, H, W]
        
        Returns
        -------
        tuple[dict[str, torch.Tensor], list[float]]
            - Dictionary with loss metrics (keys: "mse")
            - List of weights for each loss term
        """
        output_metrics = {}
        output_weights = []
        
        output_metrics["mse"] = torch.nn.functional.mse_loss(preds, target)
        output_weights.append(1)
        
        return output_metrics, output_weights
    
    def train_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate evaluation metrics for training.
        
        These metrics are for monitoring purposes only and do not contribute to loss.
        Uses MAE (Mean Absolute Error) for reporting prediction accuracy.
        
        Parameters
        ----------
        preds : torch.Tensor
            Model predictions, shape [B, H, W] or [B, 1, H, W]
        target : torch.Tensor
            Ground truth WSA maps, shape [B, H, W] or [B, 1, H, W]
        
        Returns
        -------
        tuple[dict[str, torch.Tensor], list[float]]
            - Dictionary with evaluation metrics (keys: "mae")
            - List of weights for each metric term
        """
        output_metrics = {}
        output_weights = []
        
        # Flatten for metrics computation
        preds_flat = self._flatten_for_metrics(preds)
        target_flat = self._flatten_for_metrics(target)
        
        self._ensure_device(preds_flat)
        output_metrics["mae"] = self._mae(preds_flat, target_flat)
        output_weights.append(1)
        
        return output_metrics, output_weights
    
    def val_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate metrics for validation.
        
        Uses MAE for monitoring prediction accuracy on validation set.
        
        Parameters
        ----------
        preds : torch.Tensor
            Model predictions, shape [B, H, W] or [B, 1, H, W]
        target : torch.Tensor
            Ground truth WSA maps, shape [B, H, W] or [B, 1, H, W]
        
        Returns
        -------
        tuple[dict[str, torch.Tensor], list[float]]
            - Dictionary with evaluation metrics (keys: "mae")
            - List of weights for each metric term
        """
        output_metrics = {}
        output_weights = []
        
        # Flatten for metrics computation
        preds_flat = self._flatten_for_metrics(preds)
        target_flat = self._flatten_for_metrics(target)
        
        self._ensure_device(preds_flat)
        output_metrics["mae"] = self._mae(preds_flat, target_flat)
        output_weights.append(1)
        
        return output_metrics, output_weights
    
    def __call__(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Dispatch to appropriate metrics method based on mode.
        
        Parameters
        ----------
        preds : torch.Tensor
            Model predictions
        target : torch.Tensor
            Ground truth targets
        
        Returns
        -------
        tuple[dict[str, torch.Tensor], list[float]]
            Metrics dictionary and weights list
        
        Raises
        ------
        NotImplementedError
            If mode is not one of "train_loss", "train_metrics", "val_metrics"
        """
        match self.mode.lower():
            case "train_loss":
                return self.train_loss(preds, target)
            
            case "train_metrics":
                with torch.no_grad():
                    return self.train_metrics(preds, target)
            
            case "val_metrics":
                with torch.no_grad():
                    return self.val_metrics(preds, target)
            
            case _:
                raise NotImplementedError(
                    f"{self.mode} is not implemented as a valid metric case."
                )