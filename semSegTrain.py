import torch
from torch.autograd import Variable
from torch.utils import data
import lr_scheduler
from model import  CrossEntropyLoss2d, FCN
from dataset import SSDataSet
from visualize import LinePlotter
from torchvision.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip, Resize
from PIL import Image
import sys
import progressbar


if __name__ == "__main__":

    haveCuda = torch.cuda.is_available()

    size = 128


    input_transform = Compose([
        Resize(size,Image.BILINEAR),
        ToTensor(),
        Normalize([.5, .5, .5], [.5, .5, .5]),

    ])
    target_transform = Compose([
        Resize(size,Image.NEAREST),
        ToTensor(),
    ])

    input_transform_tr = Compose([
        Resize(size, Image.BILINEAR),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([.5, .5, .5], [.5, .5, .5]),

    ])
    target_transform_tr = Compose([
        Resize(size, Image.NEAREST),
        RandomHorizontalFlip(),
        ToTensor(),
    ])

    seed = 12345678
    torch.manual_seed(seed)
    if haveCuda:
        torch.cuda.manual_seed(seed)

    batchSize = 1

    root = 'C:/data/cityscapes/' if sys.platform == 'win32' else '~/data/cityscapes'

    #sampler = torch.utils.data.sampler.SubsetRandomSampler(range(64))

    trainloader = data.DataLoader(SSDataSet(root, split="train", img_transform=input_transform_tr,
                                             label_transform=target_transform_tr), #sampler=sampler,
                                  batch_size=batchSize, shuffle=True, num_workers=2)

    valloader = data.DataLoader(SSDataSet(root, split="val", img_transform=input_transform,
                                             label_transform=target_transform), #sampler=sampler,
                                  batch_size=1, shuffle=False, num_workers=2)


    numClass = 8
    numPlanes = 32
    levels = 5
    levelDepth = 2
    kernelSize = 3

    model = FCN(numPlanes,levels,levelDepth,numClass,kernelSize,0.1)


    indices = []
    mapLoc = None if haveCuda else {'cuda:0': 'cpu'}
    if haveCuda:
        model = model.cuda()

    criterion = CrossEntropyLoss2d()

    epochs = 200
    lr = 1e-1
    weight_decay = 1e-3
    momentum = 0.5
    patience = 20

    optimizer = torch.optim.SGD( [ { 'params': model.parameters()}, ],
                                lr=lr, momentum=momentum,
                                weight_decay=weight_decay)


    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=patience,verbose=True)
    ploter = LinePlotter()

    bestLoss = 100
    bestAcc = 0
    bestIoU = 0
    bestTAcc = 0
    bestConf = torch.zeros(numClass,numClass)

    for epoch in range(epochs):

        model.eval()
        running_loss = 0.0
        running_acc = 0.0
        imgCnt = 0
        bar = progressbar.ProgressBar(0,len(trainloader),redirect_stdout=False)
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            pred = model(images)
            loss = criterion(pred,labels)

            #loss.backward()

            #optimizer.step()

            running_loss += loss.item()
            _, predClass = torch.max(pred, 1)
            labSize = predClass.size()
            running_acc += torch.sum( predClass == labels ).item()/(labSize[1]*labSize[2])*100

            bSize = images.size()[0]
            imgCnt += bSize

            bar.update(i)

        bar.finish()
        print("Epoch [%d] Training Loss: %.4f Training Pixel Acc: %.2f" % (epoch+1, running_loss/(i+1), running_acc/(imgCnt)))
        ploter.plot("loss", "train", epoch+1, running_loss/(i+1))

        running_loss = 0.0
        running_acc = 0.0
        imgCnt = 0
        conf = torch.zeros(numClass,numClass)
        IoU = torch.zeros(numClass)
        labCnts = torch.zeros(numClass)
        model.eval()
        bar = progressbar.ProgressBar(0, len(valloader), redirect_stdout=False)
        for i, (images, labels) in enumerate(valloader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            pred = model(images)
            loss = criterion(pred,labels)

            running_loss += loss.item()
            _, predClass = torch.max(pred, 1)
            labSize = predClass.size()
            running_acc += torch.sum(predClass == labels).item()/(labSize[1]*labSize[2])*100

            bSize = images.size()[0]
            imgCnt += bSize

            maskPred = torch.zeros(numClass,bSize,int(labSize[1]), int(labSize[2])).long()
            maskLabel = torch.zeros(numClass,bSize,int(labSize[1]), int(labSize[2])).long()
            for currClass in range(numClass):
                maskPred[currClass] = predClass.data == currClass
                maskLabel[currClass] = labels.data == currClass

            for labIdx in range(numClass):
                labCnts[labIdx] += torch.sum(maskLabel[labIdx]).item()
                for predIdx in range(numClass):
                    inter = torch.sum(maskPred[predIdx] & maskLabel[labIdx]).item()
                    conf[(predIdx, labIdx)] += inter
                    if labIdx == predIdx:
                        union = torch.sum(maskPred[predIdx] | maskLabel[labIdx]).item()
                        if union == 0:
                            IoU[labIdx] += 1
                        else:
                            IoU[labIdx] += inter/union

            bar.update(i)

        bar.finish()
        for labIdx in range(numClass):
            for predIdx in range(numClass):
                conf[(predIdx, labIdx)] /= (labCnts[labIdx] / 100.0)
        meanClassAcc = 0.0
        meanIoU = torch.sum(IoU/imgCnt).item() / numClass * 100
        currLoss = running_loss/(i+1)
        for j in range(numClass):
            meanClassAcc += conf[(j,j)]/numClass
        print("Epoch [%d] Validation Loss: %.4f Validation Pixel Acc: %.2f Mean Class Acc: %.2f IoU: %.2f" %
              (epoch+1, running_loss/(i+1), running_acc/(imgCnt), meanClassAcc, meanIoU))
        ploter.plot("loss", "val", epoch+1, running_loss/(i+1))

        if bestLoss > currLoss:
            conf[conf<0.001] = 0
            print(conf)
            bestConf = conf
            bestLoss = currLoss
            bestIoU = meanIoU
            bestAcc = meanClassAcc
            bestTAcc = running_acc/(imgCnt)

            torch.save(model.state_dict(), root + "bestModelSeg.pth")

        scheduler.step(currLoss)

    print("Optimization finished Validation Loss: %.4f Pixel Acc: %.2f Mean Class Acc: %.2f IoU: %.2f" % (bestLoss, bestTAcc, bestAcc, bestIoU))
    print(bestConf)

