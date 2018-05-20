import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import progressbar
import visualize
import sys

if __name__ == "__main__":

    # Plotter
    plotter = visualize.LinePlotter("CVSDemo")

    # Data path
    root = 'C:/data/' if sys.platform == 'win32' else './data'

    # dataset and network sizes
    N,D_in,H,D_out = 128,1000,100,10

    # create in and output
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    # Simple 2 layer MLP
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            return x

    # Mean Squared Error Loss
    criterion = torch.nn.MSELoss(size_average=False)

    # Subset data sampler
    #sampler = torch.utils.data.sampler.SubsetRandomSampler(range(5000))

    # Dataloader
    trainLoader = DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=16)

    # Make sure that all runs are comparable
    torch.manual_seed(1)

    # create net
    model = Net()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=4e-3)

    # Run 200 epochs
    for epoch in range(200):

        # variables for loss
        running_loss = 0.0
        correct = 0.0
        total = 0

        # Create progress bar
        bar = progressbar.ProgressBar(0, len(trainLoader), redirect_stdout=False)

        # For every minibatch:


        bar.finish()

        plotter.plot("Training", "Error", epoch, running_loss)