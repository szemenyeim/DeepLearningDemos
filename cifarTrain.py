import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import progressbar
import visualize
import densenet
import lr_scheduler
import sys

if __name__ == "__main__":
    # Makes multiple runs comparable
    torch.cuda.manual_seed(1)

    # Create visualizer
    plotter = visualize.LinePlotter("CVSDemo")

    # path to dataset
    root = 'C:/data/' if sys.platform == 'win32' else './data'

    # Data augmentation
    transform = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25,contrast=0.25,saturation=0.25,hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                             (0.24703233, 0.24348505, 0.26158768))
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                             (0.24703233, 0.24348505, 0.26158768))
    ])

    # Datasets
    trainSet = torchvision.datasets.CIFAR10(root=root, download=True,
                                            train=True, transform=transform)
    testSet = torchvision.datasets.CIFAR10(root=root, download=True,
                                           train=False, transform=transform_val)

    #sampler = torch.utils.data.sampler.SubsetRandomSampler(range(256))

    #Data loaders
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=128, #sampler=sampler,
                                              shuffle=False, num_workers=2)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=128, #sampler=sampler,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # create net
    net = densenet.DenseNet169().cuda()

    # Loss, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,
                          nesterov=True, weight_decay=1e-4)

    # Number of restarts
    numRest = 4
    # Number of epochs per restart
    numEpoch = 75
    # Cosine annealing learning rate schedules
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,numEpoch,eta_min=5e-3)


    def train(epoch):

        # variables for loss
        running_loss = 0.0
        correct = 0.0
        total = 0

        # set the network to train (for batchnorm and dropout)
        net.train()

        # Create progress bar
        bar = progressbar.ProgressBar(0, len(trainLoader), redirect_stdout=False)

        for i, data in enumerate(trainLoader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # compute statistics
            running_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).sum()

            bar.update(i)

        bar.finish()
        # print and plot statistics
        tr_loss = running_loss / i
        tr_corr = correct / total * 100
        print("Train epoch %d loss: %.3f correct: %.2f" % (epoch + 1, running_loss / i, tr_corr))
        plotter.plot("Loss", "Train", epoch,tr_loss)
        plotter.plot("Accuracy", "Train", epoch,tr_corr)

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
            inputs, labels = Variable(inputs.cuda(), volatile = True), Variable(labels.cuda(), volatile = True)

            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # compute statistics
            running_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).sum()

            bar.update(i)

        bar.finish()
        # print and plot statistics
        val_loss = running_loss / i
        val_corr = correct / total * 100
        print("Test epoch %d loss: %.3f correct: %.2f" % (epoch + 1, running_loss / i, val_corr))
        plotter.plot("Loss", "Val", epoch, val_loss)
        plotter.plot("Accuracy", "Val", epoch, val_corr)

        return val_loss, val_corr

    # Accuracies
    bestAcc = torch.zeros(numRest)

    # Restart counter
    restarts = -1

    for epoch in range(numEpoch*numRest):  # loop over the dataset multiple times

        # Every numEpoch epochs reset the sceduler
        currEp = None
        if (epoch % numEpoch) == 0:
            currEp = 0
            restarts += 1


        # Step with the scheduler
        scheduler.step(epoch=currEp)

        train(epoch)
        _,val_corr = val(epoch)

        # Save the best model
        if bestAcc[restarts] < val_corr:
            bestAcc[restarts] = val_corr
            print("Best model in restart %d, saving" % restarts)
            torch.save(net,root + ('/model%d.pth' % restarts))

    # Save the Accuracies
    torch.save(bestAcc,root+"/acc.pth")
    print('Finished Training')
