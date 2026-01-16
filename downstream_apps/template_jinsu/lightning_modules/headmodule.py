import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from .basemodule import BaseModule

# from ..models.head import MLPHead
from Surya.downstream_examples.solar_flare_forcasting.metrics import (
    DistributedClassificationMetrics as DCM,
)


class FlareDSModel(BaseModule):
    def __init__(
        self,
        model,
        optimizer_dict=None,
        scheduler_dict=None,
        eval_threshold: float = 0.5,
    ):
        super().__init__(
            optimizer_dict=optimizer_dict,
            scheduler_dict=scheduler_dict,
        )
        self.save_hyperparameters()
        self.model = model  # Just assign it
        self.evaluation_metric = DCM(threshold=eval_threshold)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:

        target = batch["label"].unsqueeze(1).float()
        output = self(batch).squeeze(-1)
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

        target = batch["label"].unsqueeze(1).float()
        output = self(batch).squeeze(-1)
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
