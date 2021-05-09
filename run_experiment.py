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
    parser.add_argument("--data_class", type=str, default="CIFAR10")
    parser.add_argument("--model_class", type=str, default="ResNet152")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # Get the data and model classese, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args() # no error with argument
    data_class = _import_class(f"ResNet_.data.{temp_args.data_class}")
    model_class = _import_class(f"ResNet_.models.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    model_group = parser.add_argument_group("Model Args")# make gruop with arguemnt
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")

    return parser

def main():
    """
    Run an experiment

    """
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"ResNet_.data.{args.data_class}")
    model_class = _import_class(f"ResNet_.models.{args.model_class}")
    data = data_class()
    model = model_class(args=args)

    lit_model_class = lit_models.BaseLitModel

    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(arg.load_checkpoint, args=args, model=model)
    else :
        lit_model = lit_model_class(args=args, model=model)
    
    logger = pl.loggers.TensorBoardLogger("../ResNet_utils/training/logs")

    args.weigths_summary = "full"
    trainer = pl.Trainer(default_root_dir='../ResNet_utils/trained_models/checkpoints').from_argparse_args(args, logger=logger)

    trainer.tune(lit_model, datamodule=data)

if __name__ == "__main__":
    main()