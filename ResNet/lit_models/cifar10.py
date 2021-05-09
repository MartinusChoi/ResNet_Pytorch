import torch
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule

from torchvision.datasets import CIFAR10 as TorchCIFAR10
from torchvision import transforms

DATA_DIR = '../ResNet_utils/data'

class LitCIFAR10(LightningDataModule):
    def __init__(self, batch_size=32, data_dir:str = DATA_DIR):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    
    def prepare_data(self):
        # download, tokenize, etc..
        TorchCIFAR10(root=self.data_dir, train=True, download=True)
        TorchCIFAR10(root=self.data_dir, train=False, download=True)
    
    def setup(self, stage: str = None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar10_full = TorchCIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [45000, 5000])
        
        # Assign test dataset for use in dataloader
        if stage == 'test' or stage is None:
            self.cifar10_test = TorchCIFAR10(root=self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, shuffle=True)