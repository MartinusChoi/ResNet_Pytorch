import argparse

import torch
import torchvision.models as models

PRETRAINED = True
PROGRESS = True

class ResNet152:

    def __init__(self, arg:argparse.Namespace = None):
        self.args = vars(args) if args is not None else {}

        self.pretrained = self.args.get("pretrained", PRETRAINED)
        self.progress = self.args.get("progress", PROGRESS)

        self.model = models.resnet152(pretrained=self.pretrained, progress=self.progress)
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrained", type=bool, help="If it's True, return model which pretrained with ImageNet.")
        parser.add_argument("--progress", type=bool)