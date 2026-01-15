import torch
import lightning as L
from transformers import get_cosine_schedule_with_warmup


class BaseModule(L.LightningModule):
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

        # Optimizer
        match self.optimizer_dict["optimizer_type"]:
            case "adam":
                optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=float(self.optimizer_dict["lr"]),
                    weight_decay=float(self.optimizer_dict["weight_decay"]),
                )
            case "adamw":
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=float(self.optimizer_dict["lr"]),
                    weight_decay=float(self.optimizer_dict["weight_decay"]),
                )
            case _:
                raise ValueError("Unknown optimizer")

        # Scheduler
        match self.scheduler_dict["scheduler_type"]:
            case "reducelronplateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.scheduler_dict["factor"],
                    patience=self.scheduler_dict["patience"],
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }

            case "cosine_with_warmup":
                num_warmup_steps = int(
                    self.scheduler_dict["warmup_ratio"] * self.total_steps
                )
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
                        "frequency": 1,
                    },
                }

            case "onecyclelr":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=float(self.optimizer_dict["lr"]),
                    total_steps=self.total_steps,
                    anneal_strategy=self.scheduler_dict["anneal_strategy"],
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": 1,
                    },
                }

            case _:
                raise ValueError("Unknown scheduler")
