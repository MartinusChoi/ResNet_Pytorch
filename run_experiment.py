import argparse
import importlib

import numpy as np
import torch
import pytorch_lightning as pl

from ResNet_ import lit_models

np.random.seed(42)
torch.manual_seed(42)

def _import_class(module_and_class_name: str) -> type:
    """
    Import class from a module,

    e.g. ResNet_.lit_modles.base
    """
    # split from backside (max_split = 1)
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def _setup_parser():
    """
    Set up python's ArgumentParser with
    
    data, model, trainer, and other arguments.
    """

    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments(--max_epochs, --gpus ...)
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--data_class", type=str, default="CIFAR10DataLoader")
    parser.add_argument("--model_class", type=str, default="ResNet152")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # Get the data and model classese, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args() # no error with argument
    data_class = _import_class(f"ResNet_.data.{temp_args.data_class}")
    model_class = _import_class(f"Resnet_.models.{temp_args.model_class}")


model = resnet152_adam.ResNet_Adam()
cifar10 = CIFAR10.CIFAR10DataLoader()

trainer = pl.Trainer()

trainer.fit(model, cifar10)