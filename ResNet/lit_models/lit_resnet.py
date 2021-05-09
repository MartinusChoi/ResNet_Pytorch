import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer

from torchvision import models

import cifar10

OPTIMZIER = "Adam"
LR = 1e-3
LOSS = "cross_entropy"

class Accuracy(pl.metrics.Accuracy):
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:

        if preds.min() < 0 or preds.max() > 1:
            preds = F.softmax(preds, dim=-1)
        super().update(preds=preds, target=target)


class LitResNetModule(pl.LightningModule):
    def __init__(self, args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.resnet = models.resnet152(pretrained=True, progress=True)
        self.params = self.resnet.parameters()

        optimizer = self.args.get("optimizer", OPTIMZIER)
        self.optimizer_class = getattr(optim, optimizer)

        self.lr = self.args.get("lr", LR)

        loss = self.args.get("loss", LOSS)
        self.loss_fn = getattr(F, loss)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMZIER, help="optimizezr class from torch.optim")
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function form torch.nn.functional")

    def forward(self, x):
        return self.resnet(x)
    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.params, lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        z = self.resnet(x)
        loss = self.loss_fn(z, y)
        self.log("train_loss", loss, on_epoch=True)
        self.train_acc(z, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.resnet(x)
        loss = self.loss_fn(z, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(z, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        z = self.resnet(x)
        self.test_acc(z, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)

if __name__ == "__main__":
    model = LitResNetModule()

    trainer = pl.Trainer()

    cifar10_dm = cifar10.LitCIFAR10()

    trainer = Trainer(gpus=1)

    trainer.fit(model, cifar10_dm)