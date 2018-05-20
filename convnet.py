import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import progressbar
import visualize
import sys

if __name__ == "__main__":

    haveCuda = torch.cuda.is_available()

    # Makes multiple runs comparable
    if haveCuda:
        torch.cuda.manual_seed(1)
    else:
        torch.manual_seed(1)

    # path to dataset
    root = 'C:/data/' if sys.platform == 'win32' else './data'

    # Create visualizer
    plotter = visualize.LinePlotter("CVSDemo")

    # Data augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    sampler = torch.utils.data.sampler.SubsetRandomSampler(range(10000))

    # Datasets
    trainSet = torchvision.datasets.CIFAR10(root=root,
        download=True, train=True, transform=transform)
    testSet = torchvision.datasets.CIFAR10(root=root,
        download=True, train=False, transform=transform_val)

    #Data loaders
    trainLoader = torch.utils.data.DataLoader(trainSet, sampler=sampler,
        batch_size=128, shuffle=False, num_workers=2)
    testLoader = torch.utils.data.DataLoader(testSet, sampler=sampler,
        batch_size=128, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # Define small network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 5)
            self.pool = nn.MaxPool2d(4, 4)
            self.conv2 = nn.Conv2d(32, 64, 7)
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = F.relu(self.conv2(x))
            x = x.view(-1, 64)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # create net
    net = Net()
    if haveCuda:
        net = net.cuda()

    # Loss, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01,
                momentum=0.5, weight_decay=1e-3)

    def train( epoch ):

        # variables for loss
        running_loss = 0.0
        correct = 0.0
        total = 0

        # set the network to train (for batchnorm and dropout)
        net.train()

        # Create progress bar
        bar = progressbar.ProgressBar(0, len(trainLoader), redirect_stdout=False)

        for i, data in enumerate(trainLoader):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if haveCuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # compute statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            bar.update(i)

        bar.finish()
        # print and plot statistics
        tr_loss = running_loss / i
        tr_corr = correct / total * 100
        print("Train epoch %d loss: %.3f correct: %.2f" % (epoch + 1, running_loss / i, tr_corr))
        plotter.plot("Loss", "Train", epoch, tr_loss)
        plotter.plot("Accuracy", "Train", epoch, tr_corr)

    def val(epoch):

        # variables for loss
        running_loss = 0.0
        correct = 0.0
        total = 0

        # set the network to eval (for batchnorm and dropout)
        net.eval()

        # Create progress bar
        bar = progressbar.ProgressBar(0, len(testLoader), redirect_stdout=False)

        for i, data in enumerate(testLoader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if haveCuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # compute statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            bar.update(i)

        bar.finish()
        # print and plot statistics
        val_loss = running_loss / i
        val_corr = correct / total * 100
        print("Test epoch %d loss: %.3f correct: %.2f" % (epoch + 1, running_loss / i, val_corr))
        plotter.plot("Loss", "Val", epoch, val_loss)
        plotter.plot("Accuracy", "Val", epoch, val_corr)

    for epoch in range(50):  # loop over the dataset multiple times
        train(epoch)
        val(epoch)

        # After 25 epochs decrease the learning rate
        if epoch == 24:
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5, weight_decay=1e-5)


    print('Finished Training')
