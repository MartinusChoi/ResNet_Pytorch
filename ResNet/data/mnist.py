import torch
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule

from torchvision.datasets import MNIST as TorchMNIST
from torchvision import transforms

DATA_DIR = '../ResNet_utils/data'

class Litmnist(LightningDataModule):
    def __init__(self, batch_size=32, data_dir:str = DATA_DIR):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))]
        )
    
    def prepare_data(self):
        TorchMNIST(root=self.data_dir, train=True, download=True)
        TorchMNIST(root=self.data_dir, train=False, download=True)
    
    def setup(self, stage:str = None):
        if stage == 'fit' or stage is None:
            mnist_full = TorchMNIST(root=self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        
        if stage == 'test' or stage is None:
            self.mnist_test = TorchMNIST(root=self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=True)