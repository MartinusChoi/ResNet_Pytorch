import argparse

import torch
from torch.optim import Adam
import pytorch_lightning as pl
from torchvision.models import resnet152

OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "cross_entropy"

class Accuracy(pl.metrics.Accuracy):
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:

        if preds.min() < 0 or preds.max() > 1:
            preds = F.softmax(preds, dim=-1)
        super().update(preds=preds, target=target)

class BaseLitModel(pl.LightningModule):

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {} # argparse.Namespase => dict

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer) # get attributes from object (optimizer from torch.optim)

        self.lr = self.args.get("lr", LR)

        loss = self.args.get("loss", LOSS)
        if loss not in ("cross_entropy"):
            self.loss_fn = getattr(torch.nn.functional, loss)
        
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="optimizezr class from torch.optim")
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function form torch.nn.functional")
    
    def configure_optimizers(self) :
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimzier
    
    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        outputs = self.model(inputs)
        loss = self.loss(outputs, labels)
        self.log("train_loss", loss)
        self.train_acc(outputs, lables)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        torch.cuda.empty_cache()
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        outputs = self.model(inputs)
        loss = self.loss(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(outputs, labels)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        torch.cuda.empty_cache()
    
    def test_step(self, test_batch, batch_idx):
        inputs, labels = test_batch
        outputs = self.model(inputs)
        self.test_acc(outputs, labels)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)