"""
Metrics for Stokes profile regression.

This module defines metrics for evaluating Stokes profile predictions,
where both predictions and targets have shape [B, 4, n_wavelengths, H, W].
"""

import torch
import torch.nn.functional as F
import torchmetrics as tm


class StokesMetrics:
    """
    Metrics class for Stokes profile regression.
    
    Handles predictions and targets with shape [B, 4, n_wavelengths, H, W],
    where:
    - B: batch size
    - 4: Stokes parameters (I, Q, U, V)
    - n_wavelengths: number of wavelength points
    - H, W: spatial dimensions
    """
    
    def __init__(self, mode: str):
        """
        Initialize StokesMetrics class.
        
        Args:
            mode (str): Mode to use for metric evaluation. Can be "train_loss",
                        "train_metrics", or "val_metrics".
        """
        self.mode = mode
        
        # Cache torchmetrics instances once
        self._mse = tm.MeanSquaredError()
        self._mae = tm.MeanAbsoluteError()
    
    def _ensure_device(self, preds: torch.Tensor):
        """Move metric modules to the same device as preds."""
        if self._mse.device != preds.device:
            self._mse = self._mse.to(preds.device)
            self._mae = self._mae.to(preds.device)
    
    def train_loss(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate loss metrics for training.
        
        Args:
            preds: Model predictions [B, 4, n_wavelengths, H, W]
            target: Ground truth [B, 4, n_wavelengths, H, W]
        
        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - Dictionary containing loss metrics
                - List of weights for each metric
        """
        output_metrics = {}
        output_weights = []
        
        # Check for NaN or inf values and replace with zeros
        preds = torch.where(torch.isfinite(preds), preds, torch.zeros_like(preds))
        target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
        
        # Mean Squared Error over all dimensions
        mse = F.mse_loss(preds, target)
        # Replace NaN with zero if it occurs
        mse = torch.where(torch.isfinite(mse), mse, torch.tensor(0.0, device=mse.device))
        output_metrics["mse"] = mse
        output_weights.append(1.0)
        
        return output_metrics, output_weights
    
    def train_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate evaluation metrics for training (reporting only).
        
        Args:
            preds: Model predictions [B, 4, n_wavelengths, H, W]
            target: Ground truth [B, 4, n_wavelengths, H, W]
        
        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - Dictionary containing evaluation metrics
                - List of weights for each metric
        """
        output_metrics = {}
        output_weights = []
        
        # Check for NaN or inf values and replace with zeros
        preds = torch.where(torch.isfinite(preds), preds, torch.zeros_like(preds))
        target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
        
        # Mean Absolute Error
        mae = F.l1_loss(preds, target)
        mae = torch.where(torch.isfinite(mae), mae, torch.tensor(0.0, device=mae.device))
        output_metrics["mae"] = mae
        output_weights.append(1.0)
        
        # Per-Stokes-parameter MSE
        for i, stokes_name in enumerate(['I', 'Q', 'U', 'V']):
            mse_stokes = F.mse_loss(preds[:, i, :, :, :], target[:, i, :, :, :])
            mse_stokes = torch.where(torch.isfinite(mse_stokes), mse_stokes, torch.tensor(0.0, device=mse_stokes.device))
            output_metrics[f"mse_{stokes_name}"] = mse_stokes
            output_weights.append(1.0)
        
        return output_metrics, output_weights
    
    def val_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate metrics for validation.
        
        Args:
            preds: Model predictions [B, 4, n_wavelengths, H, W]
            target: Ground truth [B, 4, n_wavelengths, H, W]
        
        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - Dictionary containing validation metrics
                - List of weights for each metric
        """
        output_metrics = {}
        output_weights = []
        
        # Check for NaN or inf values and replace with zeros
        preds = torch.where(torch.isfinite(preds), preds, torch.zeros_like(preds))
        target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
        
        # Overall MSE
        mse = F.mse_loss(preds, target)
        mse = torch.where(torch.isfinite(mse), mse, torch.tensor(0.0, device=mse.device))
        output_metrics["mse"] = mse
        output_weights.append(1.0)
        
        # Mean Absolute Error
        mae = F.l1_loss(preds, target)
        mae = torch.where(torch.isfinite(mae), mae, torch.tensor(0.0, device=mae.device))
        output_metrics["mae"] = mae
        output_weights.append(1.0)
        
        # Per-Stokes-parameter MSE
        for i, stokes_name in enumerate(['I', 'Q', 'U', 'V']):
            mse_stokes = F.mse_loss(preds[:, i, :, :, :], target[:, i, :, :, :])
            mse_stokes = torch.where(torch.isfinite(mse_stokes), mse_stokes, torch.tensor(0.0, device=mse_stokes.device))
            output_metrics[f"mse_{stokes_name}"] = mse_stokes
            output_weights.append(1.0)
        
        # Per-Stokes-parameter MAE
        for i, stokes_name in enumerate(['I', 'Q', 'U', 'V']):
            mae_stokes = F.l1_loss(preds[:, i, :, :, :], target[:, i, :, :, :])
            mae_stokes = torch.where(torch.isfinite(mae_stokes), mae_stokes, torch.tensor(0.0, device=mae_stokes.device))
            output_metrics[f"mae_{stokes_name}"] = mae_stokes
            output_weights.append(1.0)
        
        return output_metrics, output_weights
    
    def __call__(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Default method to evaluate all metrics.
        
        Args:
            preds: Model predictions [B, 4, n_wavelengths, H, W]
            target: Ground truth [B, 4, n_wavelengths, H, W]
        
        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - Dictionary with all metrics
                - List of weights for each metric
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
