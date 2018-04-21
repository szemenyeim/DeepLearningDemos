# CVSDemo
## CIFAR-10 Demos in PyTorch

This repo contains demos used in the Computer Vision Systems course at BUTE (BMEVIIIMA07).

## REQUIREMENTS:

* Python 3
* PyTorch 3.2 from www.pytorch.org
* torchvision `pip install torchvision`
* progressbar2 `pip install progressbar`
* visdom `pip install visdom`

The first demo uses a simple neural network training implemented in plain PyTorch using nothing but autograd. You can run this demo by:

`python net_plain.py`

The second demo trains a neural network on random data using five different learning rate settings.

`python net_torch.py`

The third demo trains a small convolutional neural network on CIFAR-10. This achieves about 70% validation accuracy.

`python convnet.py`

The fourth demo trains a DenseNet169 net on CIFAR-10. This achieves ~95.6% validation accuracy.

`python cifarTrain.py`

`python cifarTest.py`

## CREDITS:

DenseNet implementation by https://github.com/kuangliu/pytorch-cifar
