import pytorch_lightning as pl

class FlareLightningModule(pl.LightningModule):
    def __init__(self, model, metrics, lr):
        super().__init__()
        self.model = model
        self.training_loss
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["ts"]
        y = batch["forecast"].unsqueeze(1).float()

        y_hat = self(x)
        training_losses, training_loss_weights = self.metrics


        loss = self.loss_fn(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["ts"]
        y = batch["forecast"].unsqueeze(1).float()

        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)