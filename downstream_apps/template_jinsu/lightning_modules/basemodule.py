import torch
import pytorch_lightning as pl


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        scheduler_dict: dict,
        optimizer_dict: dict,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_dict = scheduler_dict
        self.optimizer_dict = optimizer_dict

    def configure_optimizers(self):

        total_steps = self.trainer.estimated_stepping_batches

        match self.optimizer_dict.optimizer_type:
            case "adam":
                optimizer = torch.optim.Adam(
                    self.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )
            case "adamw":
                optimizer = torch.optim.AdamW(
                    self.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )

        match self.scheduler_dict.scheduler_type:
            case "reducelronplateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.scheduler_params.factor,
                    patience=self.scheduler_params.patience,
                )

            case "onecyclelr":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.lr,
                    total_steps=total_steps,
                    # steps_per_epoch=self.optimizer_params.steps_per_epoch,
                    # epochs=self.optimizer_params.epochs,
                    # pct_start=self.scheduler_params.pct_start,
                    anneal_strategy=self.scheduler_params.anneal_strategy,
                )

        return [optimizer], [scheduler]
