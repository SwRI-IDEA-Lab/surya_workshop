import pytorch_lightning as pl
import torch

class FlareLightningModule(pl.LightningModule):
    def __init__(self, model, metrics, lr):
        super().__init__()
        self.model = model
        self.training_loss = metrics['train_loss']
        self.training_evaluation = metrics['train_metrics']
        self.validation_evaluation = metrics['val_metrics']
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x = batch["ts"]
        target = batch["forecast"].unsqueeze(1).float()

        output = self(x)
        training_losses, training_loss_weights = self.training_loss(output, target)

        # Combine losses according to their weights
        for n, key in enumerate(training_losses.keys()):
            if n == 0:
                loss = training_losses[key] * training_loss_weights[n]
            else:
                loss += training_losses[key] * training_loss_weights[n]

        # Add all reporting for losses
        self.log("train_loss", loss, prog_bar=True)
        for key in training_losses.keys():
            self.log(f"train_loss_{key}", training_losses[key], prog_bar=False)

        # Add all reporting for evaluation metrics
        training_evaluation_metrics, training_evaluation_weights = self.training_evaluation(output, target)
        if len(training_evaluation_weights) > 0:
            for n, key in enumerate(training_evaluation_metrics.keys()):
                self.log(f"train_metric_{key}", training_evaluation_metrics[key], prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["ts"]
        target = batch["forecast"].unsqueeze(1).float()

        output = self(x)
        val_losses, val_loss_weights = self.training_loss(output, target)

        # Combine losses according to their weights
        for n, key in enumerate(val_losses.keys()):
            if n == 0:
                loss = val_losses[key] * val_loss_weights[n]
            else:
                loss += val_losses[key] * val_loss_weights[n]

        # Add all reporting for losses
        self.log("val_loss", loss, prog_bar=True)
        for key in val_losses.keys():
            self.log(f"val_loss_{key}", val_losses[key], prog_bar=False)

        # Add all reporting for evaluation metrics
        val_evaluation_metrics, val_evaluation_weights = self.validation_evaluation(output, target)
        if len(val_evaluation_weights) > 0:
            for n, key in enumerate(val_evaluation_metrics.keys()):
                self.log(f"val_metric_{key}", val_evaluation_metrics[key], prog_bar=False)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)