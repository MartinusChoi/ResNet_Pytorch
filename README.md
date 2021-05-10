# ResNet_Pytorch

Implement ResNet pretrained py ImageNet with pytorch.

three datasets are available.
- MNIST
- Fashion-MNIST
- CIFAR10

two optimizers are available.
- Adam
- Adagrad

when you run the experiment, 
- dataset are saved in directory `Utils/data/`
- model checkpoints are saved in directory `Utils/checkpoints`
- logs are saved in directory `Utils/tb_lightning_logs`

## Run experiment

### Run with `Adam` optimizer

#### - MNIST dataset

~~~
python run_experiment.py --max_epochs 10 --data_class mnist --optimizer Adam --gpus 1
~~~

#### - Fashion-MNIST dataset

~~~
python run_experiment.py --max_epochs 10 --data_class fashionmnist --optimizer Adam --gpus 1
~~~

#### - CIFAR10 dataset

~~~
python run_experiment.py --max_epochs 10 --data_class cifar10 --optimizer Adam --gpus 1
~~~

### Run with `Adagrad` optimizer

#### - MNIST dataset

~~~
python run_experiment.py --max_epochs 10 --data_class mnist --optimizer Adagrad --gpus 1
~~~

#### - Fashion-MNIST dataset

~~~
python run_experiment.py --max_epochs 10 --data_class fashionmnist --optimizer Adagrad --gpus 1
~~~

#### - CIFAR10 dataset

~~~
python run_experiment.py --max_epochs 10 --data_class cifar10 --optimizer Adagrad --gpus 1
~~~

## Arguments

- `--data_class` : select dataset. must be one of `mnist`, `fashionmnist` and `cifar10`. default is `cifar10`
- `--optimizer` : select optimizer. must be one of `Adam` and `Adagrad`. default is `Adam`
- `--lr` : set value for learning rate. type is float. default is `1e-3`
- `--loss` : set loss metric. default is `cross_entropy`
- `--max_epochs` : set maximum epochs.
- `--gpus` : set option for gpus. if you want run with cpu, set this value with `0`.

## Check Tensorboard log

after you run experiment, you can check the log in tensorboard.

first, move to correct directory, and command like this.

~~~
tensorboard --logdir ./Utils/tb_lightning_logs
~~~

## Reference

- [fsdl-text-recognizer-2021-labs](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs) : Reference system architecture
- [torchvision master documentation, models](https://pytorch.org/vision/stable/models.html) : follow document to load ResNet pretrained with ImageNet
- [torchvision master documentation, datasets](https://pytorch.org/vision/stable/datasets.html) : follow document to load MNIST, Fashion-MNIST, CIFAR10 dataset
- [PyTorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/) : follow document to use pytorch lightning module
- [pytorch-resnet-mnist](https://github.com/marrrcin/pytorch-resnet-mnist/blob/master/pytorch-resnet-mnist.ipynb) : use `self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)` for solving channel problem of MNIST dataset
