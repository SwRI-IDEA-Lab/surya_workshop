"""
PyTorch Lightning wrapper for training a Stokes profile prediction model.

This module defines a LightningModule for Stokes profile regression that:
  - Calls a user-provided PyTorch model on batched inputs (batch["stokes_input"])
  - Computes training/validation losses via user-provided loss functions
  - Logs scalar losses and evaluation metrics
  - Configures an Adam optimizer
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import lightning as L
import torch


# Type aliases for clarity
LossDict = Mapping[str, torch.Tensor]
MetricDict = Mapping[str, torch.Tensor]
Weights = Any  # often a list[float] or list[torch.Tensor]


class StokesLightningModule(L.LightningModule):
    """
    PyTorch LightningModule for Stokes profile prediction training.
    
    This class wraps:
      (1) a user-provided PyTorch model (nn.Module-like) and
      (2) a set of loss/metric callables packaged in the `metrics` dictionary.
    
    Parameters
    ----------
    model:
        A callable model (typically torch.nn.Module) that accepts the batch dictionary
        and returns predictions with shape [B, 4, n_wavelengths, H, W].
    
    metrics:
        Dictionary containing the training loss function and metric functions.
        Required keys:
          - "train_loss": callable(output, target) -> (losses, weights)
          - "train_metrics": callable(output, target) -> (metrics, weights)
          - "val_metrics": callable(output, target) -> (metrics, weights)
    
    lr:
        Learning rate for the Adam optimizer.
    
    batch_size:
        Optional batch size passed to Lightning's `self.log(..., batch_size=...)`.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, Callable[..., Tuple[Dict[str, torch.Tensor], Weights]]],
        lr: float,
        batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.model = model
        
        # Loss callable: returns (loss_dict, weight_list)
        self.training_loss = metrics["train_loss"]
        
        # Metric callables: return (metric_dict, weight_list)
        self.training_evaluation = metrics["train_metrics"]
        self.validation_evaluation = metrics["val_metrics"]
        
        self.lr = lr
    
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass used by Lightning and by explicit calls in steps.
        
        Parameters
        ----------
        batch:
            Input batch dictionary containing 'stokes_input' key.
        
        Returns
        -------
        torch.Tensor
            Model predictions with shape [B, 4, n_wavelengths, H, W].
        """
        return self.model(batch)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Runs one training step on a single batch.
        
        Workflow:
        1) Extract targets from batch: target = batch["forecast"]
        2) Compute model output: output = self(batch)
        3) Compute losses and combine via weights
        4) Log losses and metrics
        
        Returns
        -------
        torch.Tensor
            The scalar training loss used for backpropagation.
        """
        target = batch["forecast"].float()  # [B, 4, n_wavelengths, H, W]
        
        output = self(batch)  # [B, 4, n_wavelengths, H, W]
        
        # Check for NaN in output and replace with zeros
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"Warning: NaN/Inf detected in model output at batch {batch_idx}, replacing with zeros")
            output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))
        
        # Check for NaN in target
        if torch.isnan(target).any() or torch.isinf(target).any():
            print(f"Warning: NaN/Inf detected in target at batch {batch_idx}, replacing with zeros")
            target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
        
        training_losses, training_loss_weights = self.training_loss(output, target)
        
        # Combine losses according to their weights
        loss = None
        for n, key in enumerate(training_losses.keys()):
            component = training_losses[key] * training_loss_weights[n]
            # Replace NaN in component
            component = torch.where(torch.isfinite(component), component, torch.tensor(0.0, device=component.device))
            loss = component if loss is None else (loss + component)
        
        if loss is None:
            raise ValueError("training_loss returned an empty loss dict; cannot compute scalar loss.")
        
        # Replace NaN in loss
        loss = torch.where(torch.isfinite(loss), loss, torch.tensor(0.0, device=loss.device))
        
        # Log aggregate loss and component losses
        self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size)
        for key in training_losses.keys():
            train_loss = training_losses[key]
            train_loss = torch.where(torch.isfinite(train_loss), train_loss, torch.tensor(0.0, device=train_loss.device))
            self.log(f"train_loss_{key}", train_loss, prog_bar=False, batch_size=self.batch_size)
        
        # Log evaluation metrics (optional)
        training_evaluation_metrics, training_evaluation_weights = self.training_evaluation(output, target)
        if len(training_evaluation_weights) > 0:
            for key in training_evaluation_metrics.keys():
                metric_val = training_evaluation_metrics[key]
                metric_val = torch.where(torch.isfinite(metric_val), metric_val, torch.tensor(0.0, device=metric_val.device))
                self.log(f"train_metric_{key}", metric_val, prog_bar=False, batch_size=self.batch_size)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """
        Runs one validation step on a single batch.
        
        Workflow:
        1) Extract targets
        2) Compute output
        3) Compute validation losses and combine via weights
        4) Log losses and metrics
        """
        target = batch["forecast"].float()  # [B, 4, n_wavelengths, H, W]
        
        output = self(batch)  # [B, 4, n_wavelengths, H, W]
        
        # Check for NaN in output and replace with zeros
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"Warning: NaN/Inf detected in model output at batch {batch_idx}, replacing with zeros")
            output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))
        
        # Check for NaN in target
        if torch.isnan(target).any() or torch.isinf(target).any():
            print(f"Warning: NaN/Inf detected in target at batch {batch_idx}, replacing with zeros")
            target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
        
        val_losses, val_loss_weights = self.training_loss(output, target)
        
        # Combine losses according to their weights
        loss = None
        for n, key in enumerate(val_losses.keys()):
            component = val_losses[key] * val_loss_weights[n]
            # Replace NaN in component
            component = torch.where(torch.isfinite(component), component, torch.tensor(0.0, device=component.device))
            loss = component if loss is None else (loss + component)
        
        if loss is None:
            raise ValueError("training_loss returned an empty loss dict; cannot compute scalar val loss.")
        
        # Replace NaN in loss
        loss = torch.where(torch.isfinite(loss), loss, torch.tensor(0.0, device=loss.device))
        
        # Log aggregate loss and component losses
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)
        for key in val_losses.keys():
            val_loss = val_losses[key]
            val_loss = torch.where(torch.isfinite(val_loss), val_loss, torch.tensor(0.0, device=val_loss.device))
            self.log(f"val_loss_{key}", val_loss, prog_bar=False, batch_size=self.batch_size)
        
        # Log evaluation metrics (optional)
        val_evaluation_metrics, val_evaluation_weights = self.validation_evaluation(output, target)
        if len(val_evaluation_weights) > 0:
            for key in val_evaluation_metrics.keys():
                metric_val = val_evaluation_metrics[key]
                metric_val = torch.where(torch.isfinite(metric_val), metric_val, torch.tensor(0.0, device=metric_val.device))
                self.log(f"val_metric_{key}", metric_val, prog_bar=False, batch_size=self.batch_size)
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer used by Lightning.
        
        Returns
        -------
        torch.optim.Optimizer
            Adam optimizer over all module parameters with learning rate `self.lr`.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def on_before_optimizer_step(self, optimizer):
        """Clip gradients to prevent NaN values from gradient explosion."""
        # Clip gradients to prevent NaN
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
