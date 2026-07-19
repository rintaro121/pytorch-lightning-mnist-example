import lightning as L
import torch
import torch.nn.functional as F

from model import MLP


class MNISTLightningModule(L.LightningModule):
    def __init__(
        self,
        lr=1e-3,
        hidden_dim=128,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MLP(hidden_dim=hidden_dim)

    def forward(self, x):
        return self.model(x)

    def _compute_accuracy(self, logits, y):
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        return acc

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self._compute_accuracy(logits, y)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self._compute_accuracy(logits, y)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self._compute_accuracy(logits, y)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
