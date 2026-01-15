from __future__ import annotations
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import lightning as L
import torch

LossDict = Mapping[str, torch.Tensor]
MetricDict = Mapping[str, torch.Tensor]
Weights = Any


class EVELightningModule(L.LightningModule):
    """
    LightningModule for EVE spectra regression.

    Identical to FlareLightningModule except target shaping:
    - scalar targets: (B,) -> (B,1)
    - vector targets: (B,L) -> unchanged
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

        self.training_loss = metrics["train_loss"]
        self.training_evaluation = metrics["train_metrics"]
        self.validation_evaluation = metrics["val_metrics"]

        self.lr = lr

    @staticmethod
    def _format_target(y: torch.Tensor) -> torch.Tensor:
        """
        Make targets compatible with both scalar and vector regression.
        """
        y = y.float()
        if y.ndim == 1:
            # (B,) -> (B,1)
            y = y.unsqueeze(1)
        # if (B,L) leave it as is
        return y

    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        # Note: your models (RegressionEVEModel, HelioSpectformer1D) accept the batch dict
        return self.model(x)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        target = self._format_target(batch["forecast"])
        output = self(batch)

        training_losses, training_loss_weights = self.training_loss(output, target)

        loss = None
        for n, key in enumerate(training_losses.keys()):
            component = training_losses[key] * training_loss_weights[n]
            loss = component if loss is None else (loss + component)

        if loss is None:
            raise ValueError("training_loss returned an empty loss dict; cannot compute scalar loss.")

        self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        for key in training_losses.keys():
            self.log(f"train_loss_{key}", training_losses[key], prog_bar=False, batch_size=self.batch_size, sync_dist=True)

        training_evaluation_metrics, training_evaluation_weights = self.training_evaluation(output, target)
        if len(training_evaluation_weights) > 0:
            for key in training_evaluation_metrics.keys():
                self.log(f"train_metric_{key}", training_evaluation_metrics[key], prog_bar=False, batch_size=self.batch_size, sync_dist=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        target = self._format_target(batch["forecast"])
        output = self(batch)

        val_losses, val_loss_weights = self.training_loss(output, target)

        loss = None
        for n, key in enumerate(val_losses.keys()):
            component = val_losses[key] * val_loss_weights[n]
            loss = component if loss is None else (loss + component)

        if loss is None:
            raise ValueError("training_loss returned an empty loss dict; cannot compute scalar val loss.")

        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        for key in val_losses.keys():
            self.log(f"val_loss_{key}", val_losses[key], prog_bar=False, batch_size=self.batch_size, sync_dist=True)

        val_evaluation_metrics, val_evaluation_weights = self.validation_evaluation(output, target)
        if len(val_evaluation_weights) > 0:
            for key in val_evaluation_metrics.keys():
                self.log(f"val_metric_{key}", val_evaluation_metrics[key], prog_bar=False, batch_size=self.batch_size, sync_dist=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
