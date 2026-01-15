import torch
import pytorch_lightning as pl
from transfomer import get_cosine_schedule_with_warmup
from loguru import logger as lgr_logger


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        scheduler_dict: dict,
        optimizer_dict: dict,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.scheduler_dict = scheduler_dict
        self.optimizer_dict = optimizer_dict

    def configure_optimizers(self):

        self.total_steps = self.trainer.estimated_stepping_batches
        lgr_logger.info(f"total_steps: {self.total_steps}")

        match self.optimizer_dict.optimizer_type:
            case "adam":
                optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=self.optimizer_dict.lr,
                    weight_decay=self.optimizer_dict.weight_decay,
                )
            case "adamw":
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.optimizer_dict.lr,
                    weight_decay=self.optimizer_dict.weight_decay,
                )

        match self.scheduler_dict.scheduler_type:
            case "reducelronplateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.scheduler_params.factor,
                    patience=self.scheduler_params.patience,
                )

                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",  # REQUIRED
                    },
                }

            case "cosine_with_warmup":
                num_warmup_steps = self.scheduler_dict.warmup_ratio * self.total_steps
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=self.total_steps,
                )

                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                    },
                }

            case "onecyclelr":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.optimizer_dict.lr,
                    total_steps=self.total_steps,
                    # steps_per_epoch=self.optimizer_params.steps_per_epoch,
                    # epochs=self.optimizer_params.epochs,
                    # pct_start=self.scheduler_params.pct_start,
                    anneal_strategy=self.scheduler_params.anneal_strategy,
                )

                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                    },
                }
