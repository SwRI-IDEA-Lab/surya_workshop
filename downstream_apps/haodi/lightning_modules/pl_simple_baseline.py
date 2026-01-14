"""
pl_simple_baseline.py

A minimal PyTorch Lightning wrapper for training a flare prediction model.

This module defines a single LightningModule (FlareLightningModule) that:
  - Calls a user-provided PyTorch model on batched inputs (batch["ts"])
  - Computes one or more training/validation losses via a user-provided loss function
  - Logs scalar losses and evaluation metrics using Lightning's built-in logging
  - Configures a simple Adam optimizer

Intended use:
  - Provide a clean, readable baseline training loop in Lightning
  - Separate "model architecture" from "training mechanics"
  - Demonstrate how to log multiple losses/metrics consistently

Key batch contract:
  - batch["ts"]       : torch.Tensor input stack (e.g., [B, C, T, H, W])
  - batch["forecast"] : torch.Tensor target values (e.g., [B] or [B,])

Key metrics contract (the `metrics` dict passed to __init__):
  - metrics["train_loss"]    : callable(output, target) -> (loss_dict, weight_list)
  - metrics["train_metrics"] : callable(output, target) -> (metric_dict, weight_list)
  - metrics["val_metrics"]   : callable(output, target) -> (metric_dict, weight_list)

Where:
  - loss_dict / metric_dict map string names -> torch scalar tensors
  - weight_list is a list-like of floats (or tensors) aligned with the dict iteration order
    used by this baseline to form a weighted sum loss.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import lightning as L
import torch


# Type aliases for clarity in documentation / teaching.
LossDict = Mapping[str, torch.Tensor]
MetricDict = Mapping[str, torch.Tensor]
Weights = Any  # often a list[float] or list[torch.Tensor]


# working on this. Turn caiik to surya stacks
class FlareLightningModule(L.LightningModule):
    """
    PyTorch LightningModule for flare prediction training.

    This class wraps:
      (1) a user-provided PyTorch model (nn.Module-like) and
      (2) a set of loss/metric callables packaged in the `metrics` dictionary.

    Parameters
    ----------
    model:
        A callable model (typically torch.nn.Module) that accepts the batch input tensor
        `x = batch["ts"]` and returns predictions `output`.

    metrics:
        Dictionary containing the training loss function and metric functions.

        Required keys:
          - "train_loss": callable(output, target) -> (losses, weights)
              losses: dict[str, torch.Tensor] scalar losses
              weights: list-like aligned with iteration order of losses.keys()
          - "train_metrics": callable(output, target) -> (metrics, weights)
          - "val_metrics": callable(output, target) -> (metrics, weights)

        The module uses:
          - train_loss for both training_step and validation_step loss computation
            (mirroring the original baseline behavior).
          - train_metrics logged during training_step (if weights is non-empty)
          - val_metrics logged during validation_step (if weights is non-empty)

    lr:
        Learning rate for the Adam optimizer.

    batch_size:
        Optional batch size passed to Lightning's `self.log(..., batch_size=...)`.
        This improves correct averaging behavior when using distributed settings
        or variable batch sizes.
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

        # create a conv layer converting input caiik to 13 channel ## 1-13-2026
        self.caiik_to_surya_layer = torch.nn.Conv2d(
            in_channels=1,
            out_channels=13,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Loss callable: returns (loss_dict, weight_list)
        self.training_loss = metrics["train_loss"]

        # Metric callables: return (metric_dict, weight_list)
        self.training_evaluation = metrics["train_metrics"]
        self.validation_evaluation = metrics["val_metrics"]

        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass used by Lightning and by explicit calls in steps.

        Parameters
        ----------
        x:
            Input tensor, typically batch["ts"].

        Returns
        -------
        torch.Tensor
            Model predictions for the batch.
        """
        
        # # 1. Create synthetic surya stack from caiik stack
        # # x['caiik'] shape: (Batch, 1, Height, Width)
        # # surya_stack shape: (Batch, Channels, Height, Width)
        # surya_stack = self.caiik_to_surya_layer(x['caiik'])

        # # 2. Add the missing Time dimension (T=1) at index 2
        # # New shape: (Batch, Channels, 1, Height, Width)
        # surya_stack = surya_stack.unsqueeze(2)

        caiik = x["caiik"]  # could be (1,H,W) or (B,1,H,W) or (B,H,W)

        if caiik.ndim == 2:                # (H,W)
            caiik = caiik.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        elif caiik.ndim == 3:
            if caiik.shape[0] == 1 and caiik.shape[1] == 4096:
                # likely (C,H,W) with C=1
                caiik = caiik.unsqueeze(0)            # (1,1,H,W)
            else:
                # likely (B,H,W)
                caiik = caiik.unsqueeze(1)            # (B,1,H,W)

        surya_stack = self.caiik_to_surya_layer(caiik)  # (B,13,H,W)
        surya_stack = surya_stack.unsqueeze(2)          # (B,13,1,H,W)

        td = x["time_delta_input"]
        # if td is (B,) make it (B,1)
        if td.ndim == 1:
            td = td.unsqueeze(1)
        # if td is (B,T) but you are forcing T=1, slice it
        if td.ndim >= 2 and td.shape[1] != 1:
            td = td[:, :1].contiguous()

        input_dict = {"ts": surya_stack, "time_delta_input": td}

        output = self.model(input_dict)

        # before RuntimeError: The size of tensor a (1280) must match the size of tensor b (2) at non-singleton dimension 1, 8pm
        # # Aggregation: If model returns a sequence (B, L, 1) or (B, L), reduce to (B, 1)  
        # if output.ndim == 3:
        #     output = output.mean(dim=1)
        # elif output.ndim == 2 and output.shape[1] > 1:
        #     output = output.mean(dim=1, keepdim=True)

        # after:
        # If the backbone returns a dict, pick the main prediction tensor, 8pm, 1/13/2026
        if isinstance(output, dict):
            # common keys; adjust if your model uses a different one
            for k in ["pred", "prediction", "forecast", "y_hat", "output", "logits"]:
                if k in output and torch.is_tensor(output[k]):
                    output = output[k]
                    break
            else:
                # fallback: first tensor value in dict
                for v in output.values():
                    if torch.is_tensor(v):
                        output = v
                        break

        # Now enforce a scalar prediction per sample: (B, 1)
        if output.ndim == 3:
            # (B, L, D) -> average over L, keep D
            output = output.mean(dim=1)
        if output.ndim == 2 and output.shape[1] != 1:
            # (B, L) -> average L, keep (B,1)
            output = output.mean(dim=1, keepdim=True)
        elif output.ndim == 1:
            output = output.unsqueeze(1)

        return output

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Runs one training step on a single batch.

        Workflow
        --------
        1) Extract inputs and targets from the batch:
              x = batch["ts"]
              target = batch["forecast"]
        2) Compute model output:
              output = self(x)
        3) Compute per-component losses and combine via provided weights:
              training_losses, training_loss_weights = training_loss(output, target)
        4) Log:
              - total weighted loss as "train_loss" (progress bar)
              - each component loss as "train_loss_<name>"
              - training metrics as "train_metric_<name>" (if any)

        Notes
        -----
        - Targets are reshaped to shape [B, 1] by unsqueeze(1) to match a common
          "single output per sample" convention.
        - The loss combination depends on dict iteration order; ensure loss dict
          insertion order is consistent if that matters.

        Returns
        -------
        torch.Tensor
            The scalar training loss used for backpropagation.
        """
        target = batch["eve_13p5"].unsqueeze(1).float()

        output = self(batch)
        training_losses, training_loss_weights = self.training_loss(output, target)

        # Combine losses according to their weights.
        # Assumes training_loss_weights aligns with iteration order of training_losses.keys().
        loss = None
        for n, key in enumerate(training_losses.keys()):
            component = training_losses[key] * training_loss_weights[n]
            loss = component if loss is None else (loss + component)

        # Safety: if no losses returned, raise a clear error.
        if loss is None:
            raise ValueError("training_loss returned an empty loss dict; cannot compute scalar loss.")

        # Log aggregate loss and component losses.
        self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size)
        for key in training_losses.keys():
            self.log(f"train_loss_{key}", training_losses[key], prog_bar=False, batch_size=self.batch_size)

        # Log evaluation metrics (optional).
        training_evaluation_metrics, training_evaluation_weights = self.training_evaluation(output, target)
        if len(training_evaluation_weights) > 0:
            for key in training_evaluation_metrics.keys():
                self.log(f"train_metric_{key}", training_evaluation_metrics[key], prog_bar=False, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """
        Runs one validation step on a single batch.

        Workflow
        --------
        1) Extract inputs and targets
        2) Compute output
        3) Compute validation losses and combine via weights
        4) Log:
              - total weighted loss as "val_loss" (progress bar)
              - each component loss as "val_loss_<name>"
              - validation metrics as "val_metric_<name>" (if any)

        Notes
        -----
        - This baseline uses `self.training_loss` to compute validation loss as well,
          matching the original code. If you need distinct train/val losses, introduce
          a separate callable (e.g., metrics["val_loss"]).
        - No value is returned (Lightning uses logs for validation tracking).
        """
        target = batch["eve_13p5"].unsqueeze(1).float()

        output = self(batch)
        val_losses, val_loss_weights = self.training_loss(output, target)

        # Combine losses according to their weights.
        loss = None
        for n, key in enumerate(val_losses.keys()):
            component = val_losses[key] * val_loss_weights[n]
            loss = component if loss is None else (loss + component)

        if loss is None:
            raise ValueError("training_loss returned an empty loss dict; cannot compute scalar val loss.")

        # Log aggregate loss and component losses.
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)
        for key in val_losses.keys():
            self.log(f"val_loss_{key}", val_losses[key], prog_bar=False, batch_size=self.batch_size)

        # Log evaluation metrics (optional).
        val_evaluation_metrics, val_evaluation_weights = self.validation_evaluation(output, target)
        if len(val_evaluation_weights) > 0:
            for key in val_evaluation_metrics.keys():
                self.log(f"val_metric_{key}", val_evaluation_metrics[key], prog_bar=False, batch_size=self.batch_size)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer used by Lightning.

        Returns
        -------
        torch.optim.Optimizer
            Adam optimizer over all module parameters with learning rate `self.lr`.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
