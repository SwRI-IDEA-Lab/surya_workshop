import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from .basemodule import BaseModule
from ..models.head import MLPHead
from Surya.downstream_examples.solar_flare_forcasting.metrics import (
    DistributedClassificationMetrics as DCM,
)


class FlareDSModel(BaseModule):
    def __init__(
        self,
        backbone,
        optimizer_dict=None,
        scheduler_dict=None,
        eval_threshold: float = 0.5,
        hidden_channels: list[int] | None = None,
        dropout: float = 0.5,
        # REMOVED: freeze_backbone argument
    ):
        super().__init__(
            optimizer_dict=optimizer_dict,
            scheduler_dict=scheduler_dict,
        )
        self.save_hyperparameters()
        self.backbone = backbone  # Just assign it
        self.evaluation_metric = DCM(threshold=eval_threshold)
        self.sigmoid = nn.Sigmoid()

        in_channels = backbone.embed_dim
        self.model = MLPHead(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token = self.backbone(x)
        output = self.model(token)
        return output

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:

        # x = batch["ts"]
        target = batch["label"].unsqueeze(1).float()
        output = self(batch)
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

        x = batch["ts"]
        target = batch["label"].unsqueeze(1).float()

        output = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)

        # evalation metic updates
        self.evaluation_metric.update(self.sigmoid(output), target)

        # Log aggregate loss and component losses.
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:

        classifier_result = self.evaluation_metric.compute_and_reset()

        for key in classifier_result.keys():
            self.log(
                f"valid/{key}",
                classifier_result[key],
                prog_bar=False,
                on_epoch=True,
                sync_dist=True,
            )
