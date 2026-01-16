"""
wsa_lightning_module.py

PyTorch Lightning module for WSA (Wang-Sheeley-Arge) map regression training.

This module is specialized for spatial 2D map outputs:
  - Input: batch["ts"] [B, C, T, H, W]
  - Target: batch["wsa_map"] [B, 1, H, W] (already 2D, no unsqueezing needed)
  - Output: model predictions [B, 1, H, W]

Handles logging of losses and metrics for training and validation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import pytorch_lightning as pl
import torch


# Type aliases for clarity
LossDict = Mapping[str, torch.Tensor]
MetricDict = Mapping[str, torch.Tensor]
Weights = Any


# class WSALightningModule(pl.LightningModule):
#     """
#     PyTorch LightningModule for WSA map prediction training.
    
#     Wraps:
#       (1) A WSA model (nn.Module) that outputs [B, 1, H, W]
#       (2) Loss and metric callables for spatial map regression
    
#     Handles:
#       - Forward pass through model
#       - Loss computation and backpropagation
#       - Metric logging during training and validation
#       - Optimizer configuration
    
#     Parameters
#     ----------
#     model : torch.nn.Module
#         WSA model (e.g., WSAModel from wsa_model_head.py)
#         Takes input [B, C, T, H, W] and outputs [B, 1, H, W]
    
#     metrics : dict
#         Dictionary with loss and metric functions.
        
#         Required keys:
#           - "train_loss": callable(output, target) -> (loss_dict, weights)
#           - "train_metrics": callable(output, target) -> (metric_dict, weights)
#           - "val_metrics": callable(output, target) -> (metric_dict, weights)
        
#         Each callable returns:
#           - loss_dict/metric_dict: dict[str, torch.Tensor] with scalar values
#           - weights: list[float] aligned with dict keys for weighted combination
    
#     lr : float
#         Learning rate for Adam optimizer
    
#     batch_size : int, optional
#         Batch size for logging (improves averaging in distributed settings)
#     """
    
#     def __init__(
#         self,
#         model: torch.nn.Module,
#         metrics: Dict[str, Callable[..., Tuple[Dict[str, torch.Tensor], Weights]]],
#         lr: float,
#         batch_size: Optional[int] = None,
#     ):
#         super().__init__()
#         self.batch_size = batch_size
#         self.model = model
        
#         # Loss and metric callables
#         self.training_loss = metrics["train_loss"]
#         self.training_evaluation = metrics["train_metrics"]
#         self.validation_evaluation = metrics["val_metrics"]
        
#         self.lr = lr
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass through the model.
        
#         Parameters
#         ----------
#         x : torch.Tensor
#             Input batch [B, C, T, H, W]
        
#         Returns
#         -------
#         torch.Tensor
#             WSA map predictions [B, 1, H, W]
#         """
#         # FIX: The HelioSpectFormer model expects a dictionary with key "ts",
#         # but the DataLoader provides a raw Tensor. We wrap it here.
#         if isinstance(x, torch.Tensor):
#             return self.model({"ts": x,"time_delta_input": 0.0})
        
#         # If it's already a dict (rare case in your setup), pass it as is
#         return self.model(x)
    
#     def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
#         """
#         Executes one training step on a batch.
        
#         Workflow:
#         1. Extract inputs and targets: x = batch["ts"], target = batch["wsa_map"]
#         2. Compute model output: output = self(x)
#         3. Compute losses and combine with weights
#         4. Log aggregate loss and component losses
#         5. Log training metrics
        
#         Parameters
#         ----------
#         batch : dict
#             Batch dictionary with keys:
#               - "ts": input tensor [B, C, T, H, W]
#               - "wsa_map": target WSA map [B, 1, H, W]
#         batch_idx : int
#             Index of batch (unused, required by Lightning)
        
#         Returns
#         -------
#         torch.Tensor
#             Scalar loss for backpropagation
#         """
#         x = batch["ts"]
#         target = batch["wsa_map"].float()  # [B, 1, H, W], no unsqueeze needed
        
#         # Forward pass
#         output = self(x)
        
#         # Compute training loss
#         training_losses, training_loss_weights = self.training_loss(output, target)
        
#         # Combine losses with weights
#         loss = None
#         for n, key in enumerate(training_losses.keys()):
#             component = training_losses[key] * training_loss_weights[n]
#             loss = component if loss is None else (loss + component)
        
#         if loss is None:
#             raise ValueError("training_loss returned empty loss dict; cannot compute scalar loss.")
        
#         # Log aggregate loss
#         self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size)
        
#         # Log component losses
#         for key in training_losses.keys():
#             self.log(
#                 f"train_loss_{key}",
#                 training_losses[key],
#                 prog_bar=False,
#                 batch_size=self.batch_size
#             )
        
#         # Log training metrics
#         training_evaluation_metrics, training_evaluation_weights = self.training_evaluation(output, target)
#         if len(training_evaluation_weights) > 0:
#             for key in training_evaluation_metrics.keys():
#                 self.log(
#                     f"train_metric_{key}",
#                     training_evaluation_metrics[key],
#                     prog_bar=False,
#                     batch_size=self.batch_size
#                 )
        
#         return loss
    
#     def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
#         """
#         Executes one validation step on a batch.
        
#         Same as training_step but:
#         - Does not compute gradients
#         - Logs to "val_loss" and "val_metric_*"
#         - Does not return loss (Lightning handles validation metrics)
        
#         Parameters
#         ----------
#         batch : dict
#             Batch dictionary (same as training_step)
#         batch_idx : int
#             Index of batch (unused, required by Lightning)
#         """
#         x = batch["ts"]
#         target = batch["wsa_map"].float()  # [B, 1, H, W]
        
#         # Forward pass
#         output = self(x)
        
#         # Compute validation loss
#         val_losses, val_loss_weights = self.training_loss(output, target)
        
#         # Combine losses with weights
#         loss = None
#         for n, key in enumerate(val_losses.keys()):
#             component = val_losses[key] * val_loss_weights[n]
#             loss = component if loss is None else (loss + component)
        
#         if loss is None:
#             raise ValueError("training_loss returned empty loss dict; cannot compute scalar val loss.")
        
#         # Log aggregate loss
#         self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)
        
#         # Log component losses
#         for key in val_losses.keys():
#             self.log(
#                 f"val_loss_{key}",
#                 val_losses[key],
#                 prog_bar=False,
#                 batch_size=self.batch_size
#             )
        
#         # Log validation metrics
#         val_evaluation_metrics, val_evaluation_weights = self.validation_evaluation(output, target)
#         if len(val_evaluation_weights) > 0:
#             for key in val_evaluation_metrics.keys():
#                 self.log(
#                     f"val_metric_{key}",
#                     val_evaluation_metrics[key],
#                     prog_bar=False,
#                     batch_size=self.batch_size
#                 )
    
#     def configure_optimizers(self) -> torch.optim.Optimizer:
#         """
#         Configure the optimizer.
        
#         Returns
#         -------
#         torch.optim.Optimizer
#             Adam optimizer with configured learning rate
#         """
#         return torch.optim.Adam(self.parameters(), lr=self.lr)


class WSALightningModule(pl.LightningModule):
    """
    PyTorch LightningModule for WSA map prediction training.
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
        
        # Loss and metric callables
        self.training_loss = metrics["train_loss"]
        self.training_evaluation = metrics["train_metrics"]
        self.validation_evaluation = metrics["val_metrics"]
        
        self.lr = lr
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Wrap input for HelioSpectFormer
        if isinstance(x, torch.Tensor):
            return self.model({"ts": x, "time_delta_input": 0.0})
        return self.model(x)

    def _sanitize_inputs(self, output: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Masks NaNs and off-disk regions to prevent NaN losses.
        
        Strategy:
        1. Identify invalid pixels in TARGET (NaNs or Zeros).
        2. Set those pixels to 0.0 in BOTH target and output.
        3. Result: Loss becomes (0.0 - 0.0) = 0.0 for those pixels.
        """
        # 1. Define Mask: Valid where target is NOT NaN AND not empty space (0.0)
        # Note: If your valid data contains actual 0.0s, remove the '> 0.0' check.
        # But for Solar Wind Speed, 0.0 usually means "background".
        valid_mask = ~torch.isnan(target) & (target > 0.0)
        
        # 2. Clone to avoid in-place errors with autograd
        clean_target = target.clone()
        clean_output = output.clone()
        
        # 3. Force invalid regions to 0.0
        # ~valid_mask selects the "bad" pixels
        clean_target[~valid_mask] = 0.0
        clean_output[~valid_mask] = 0.0
        
        return clean_output, clean_target

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        x = batch["ts"]
        target = batch["wsa_map"].float()
        
        # Forward pass
        output = self(x)
        
        # --- FIX: Sanitize before Loss ---
        clean_output, clean_target = self._sanitize_inputs(output, target)
        
        # Compute training loss using CLEANED inputs
        training_losses, training_loss_weights = self.training_loss(clean_output, clean_target)
        
        # Combine losses with weights
        loss = None
        for n, key in enumerate(training_losses.keys()):
            component = training_losses[key] * training_loss_weights[n]
            loss = component if loss is None else (loss + component)
        
        if loss is None:
            raise ValueError("training_loss returned empty loss dict; cannot compute scalar loss.")
        
        # Log aggregate loss
        self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size)
        
        # Log component losses
        for key in training_losses.keys():
            self.log(
                f"train_loss_{key}",
                training_losses[key],
                prog_bar=False,
                batch_size=self.batch_size
            )
        
        # Log training metrics (Also use cleaned inputs for accuracy metrics)
        training_evaluation_metrics, training_evaluation_weights = self.training_evaluation(clean_output, clean_target)
        if len(training_evaluation_weights) > 0:
            for key in training_evaluation_metrics.keys():
                self.log(
                    f"train_metric_{key}",
                    training_evaluation_metrics[key],
                    prog_bar=False,
                    batch_size=self.batch_size
                )
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        x = batch["ts"]
        target = batch["wsa_map"].float()
        
        # Forward pass
        output = self(x)
        
        # --- FIX: Sanitize before Loss ---
        clean_output, clean_target = self._sanitize_inputs(output, target)
        
        # Compute validation loss
        val_losses, val_loss_weights = self.training_loss(clean_output, clean_target)
        
        # Combine losses with weights
        loss = None
        for n, key in enumerate(val_losses.keys()):
            component = val_losses[key] * val_loss_weights[n]
            loss = component if loss is None else (loss + component)
        
        if loss is None:
            raise ValueError("training_loss returned empty loss dict; cannot compute scalar val loss.")
        
        # Log aggregate loss
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)
        
        # Log component losses
        for key in val_losses.keys():
            self.log(
                f"val_loss_{key}",
                val_losses[key],
                prog_bar=False,
                batch_size=self.batch_size
            )
        
        # Log validation metrics
        val_evaluation_metrics, val_evaluation_weights = self.validation_evaluation(clean_output, clean_target)
        if len(val_evaluation_weights) > 0:
            for key in val_evaluation_metrics.keys():
                self.log(
                    f"val_metric_{key}",
                    val_evaluation_metrics[key],
                    prog_bar=False,
                    batch_size=self.batch_size
                )
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)