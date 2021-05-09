import torch
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10 as TorchCIFAR10
from torchvision import transforms

class CIFAR10(pl.LightningDataModule):

    def __init__(self, batch_size=32, data_path:str = '../ResNet_utils/data'):
        self.batch_size = batch_size
        self.data_path = data_path
    
    def prepare_data(self):
        # download, tokenize, etc..
        TorchCIFAR10(root=self.data_path, train=True, download=True)
        TorchCIFAR10(root=self.data_path, train=False, download=True)
    
    def setup(self):
        # split, transforms, etc...
        # transform
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
        )

        cifar10_full = TorchCIFAR10(self.data_path, train=True, transform=transform)

        # split dataset
        self.cifar10_train, self.cifar10_train = random_split(cifar10_full, [45000, 5000])
        self.cifar10_test = TorchCIFAR10(root=self.data_path, train=False, transform=transform)
    
    def train_dataloader(self):
        cifar10_train = DataLoader(
            self.cifar10_train, batch_size=self.batch_size,
            shuffle=True, num_workers=2
        )
        return cifar10_train
    
    def val_dataloader(self):
        cifar10_val = DataLoader(
            self.cifar10_val, batch_size=self.batch_size,
            shuffle=True, num_workers=2
        )
        return cifar10_val
    
    def test_dataloader(self):
        cifar10_test = DataLoader(
            self.cifar10_test, batch_size=self.batch_size,
            shuffle=True, num_workers=2
        )
        return cifar10_test

if __name__ == "__main__":
    load_and_print_info(CIFAR10)