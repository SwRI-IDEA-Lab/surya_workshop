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

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.ops import MLP
from Surya.downstream_examples.solar_flare_forcasting.metrics import (
    DistributedClassificationMetrics as DCM,
)

from .basemodule import BaseModule

# Type aliases for clarity in documentation / teaching.
LossDict = Mapping[str, torch.Tensor]
MetricDict = Mapping[str, torch.Tensor]
Weights = Any  # often a list[float] or list[torch.Tensor]


class simple_baseline(nn.Module):
    def __init__(
        self,
        in_channels: int = 52,
        hidden_channels: list[int] = [52, 26, 1],
        dropout: float = 0.5,
    ):
        super(simple_baseline, self).__init__()
        self.model = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            activation_layer=nn.ReLU,
            norm_layer=nn.BatchNorm1d,
            dropout=dropout,
        )

    def forward(self, x):
        return self.model(torch.sigmoid(x))


class FlareBaseLine(BaseModule):
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
        optimizer_dict=None,
        scheduler_dict=None,
        batch_size: int = 1,
        eval_threshold: float = 0.5,
        in_channels: int = 52,
        hidden_channels: list[int] = [52, 26, 1],
        dropout: float = 0.5,
        # model: torch.nn.Module,
        # metrics: Dict[str, Callable[..., Tuple[Dict[str, torch.Tensor], Weights]]],
        # lr: float,
        # batch_size: Optional[int] = None,
    ):
        super().__init__(optimizer_dict=optimizer_dict, scheduler_dict=scheduler_dict)
        self.batch_size = batch_size
        self.eval_threshold = eval_threshold
        self.evaluation_metric = DCM(threshold=0.5)
        self.save_hyperparameters()

        self.model = simple_baseline(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )

        # self.batch_size = batch_size
        # self.model = model

        # # Loss callable: returns (loss_dict, weight_list)
        # self.training_loss = metrics["train_loss"]

        # # Metric callables: return (metric_dict, weight_list)
        # self.training_evaluation = metrics["train_metrics"]
        # self.validation_evaluation = metrics["val_metrics"]

        # self.lr = lr

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
        x_mean = x.mean(dim=[2, 3, 4])
        x_min = torch.amin(x, dim=[2, 3, 4])
        x_max = torch.amax(x, dim=[2, 3, 4])
        x_std = x.std(dim=[2, 3, 4])
        x_feature = torch.cat([x_mean, x_min, x_max, x_std], dim=1)
        x_feature = x_feature.view(x_feature.shape[0], -1)  # [batch, 52]

        return self.model(x_feature)

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
        x = batch["ts"]
        target = batch["label"].unsqueeze(1).float()

        # [batch, channel: 13, timestamps: 1, height: 4096, width: 4096]
        # compute statics from each channel, such as min, max, mean, std.
        # 13 channels x 4 statics = 52 features
        # [batch, channel, timetstamps, h, w] --> [batch, 13, 4]
        # flatten tensor [batch, 13, 4] --> [batch, 52]
        # feed [batch, 52] into baseline model!

        output = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)

        # Log aggregate loss and component losses.
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            sync_dist=True,
        )
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
        x = batch["ts"]
        target = batch["label"].unsqueeze(1).float()

        output = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)

        # evalation metic updates
        self.evaluation_metric.update(torch.sigmoid(output), target)

        # Log aggregate loss and component losses.
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )

    def validation_epoch_end(self) -> None:

        classifier_result = self.evaluation_metric.compute_and_reset()

        for key in self.evaluation_metric.keys():
            self.log(
                f"valid/{key}",
                classifier_result[key],
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )
