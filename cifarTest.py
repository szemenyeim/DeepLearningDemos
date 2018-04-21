import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import progressbar
import visualize
import densenet
import sys


if __name__ == "__main__":

    haveCuda = torch.cuda.is_available()

    # Makes multiple runs comparable
    if haveCuda:
        torch.cuda.manual_seed(1)
    else:
        torch.manual_seed(1)

    modelNum = 4

    # Create visualizer
    plotter = visualize.LinePlotter("CVSDemo")

    # path to dataset
    root = 'C:/data/' if sys.platform == 'win32' else './data'

    # Data transformation
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                             (0.24703233, 0.24348505, 0.26158768))
    ])

    # Dataset
    testSet = torchvision.datasets.CIFAR10(root=root, download=True,
                                           train=False, transform=transform_val)

    #Data loader
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=128,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Load accuracies: 93.81 94.54 94.88 94.84
    bestAcc = torch.load(root+"/acc.pth")

    # Load trained networks
    net = []
    for i in range(modelNum):
        net.append( densenet.DenseNet169().cuda() if haveCuda else densenet.DenseNet169() )
        state_dict = torch.load(root+ ("/model%d.pth" % i)).state_dict()
        net[i].load_state_dict(state_dict)
        net[i].eval()

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Loss variables
    running_loss = 0.0
    correct = 0.0
    total = 0

    # Set net to eval mode

    # Create progress bar
    bar = progressbar.ProgressBar(0, len(testLoader), redirect_stdout=False)

    for i, data in enumerate(testLoader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if haveCuda:
            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
        else:
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

        # Forward all networks, compute weighted average
        outputs = bestAcc[0]*net[0](inputs)
        for j in range(modelNum-1):
            outputs += bestAcc[j+1]*net[j+1](inputs)
        outputs /= torch.sum(bestAcc)

        # Get loss
        loss = criterion(outputs, labels)

        # Compute statistics
        running_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).sum()

        bar.update(i)

    bar.finish()

    # Print statistics
    val_loss = running_loss / i
    val_corr = correct / total * 100
    print("Test loss: %.3f correct: %.2f" % ( running_loss / i, val_corr))
