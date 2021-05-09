import argparse
import importlib

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from ResNet import lit_models

np.random.seed(42)
torch.manual_seed(42)

MODEL_DIR = '../ResNet_utils/trained_models/checkpoints'

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
    parser.add_argument("--data_class", type=str, default="cifar10")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # Get the data classese, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args() # no error with argument
    data_class = _import_class(f"ResNet.data.Lit{temp_args.data_class}")

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.LitResNetModule.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")

    return parser

def main():
    """
    Run an experiment

    """
    parser = _setup_parser()

    args = parser.parse_args()

    tb_logger = pl_loggers.TensorBoardLogger(MODEL_DIR)

    data_class = _import_class(f"ResNet.data.Lit{args.data_class}")
    datamodule = data_class()

    lit_model_class = lit_models.LitResNetModule

    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(arg.load_checkpoint, args=args)
    else :
        lit_model = lit_model_class(args=args)

    trainer = Trainer(default_root_dir=MODEL_DIR, logger=tb_logger).from_argparse_args(args)

    trainer.fit(lit_model, datamodule)
    trainer.test(lit_model, datamodule)

if __name__ == "__main__":
    main()