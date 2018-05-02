import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Pixelwise Cross-entropy loss
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class ConvPool(nn.Module):
    def __init__(self, inplanes, planes, dropout, num_conv=1):
        super(ConvPool, self).__init__()
        self.relu = nn.ReLU()
        self.num_conv = num_conv
        self.conv = [nn.Conv2d(inplanes, planes, kernel_size=3, dilation=2,
                              padding=2, bias=False)]
        for i in range(self.num_conv-1):
            self.conv.append( nn.Conv2d(planes, planes, kernel_size=3, dilation=2,
                              padding=2, bias=False) )
        self.pool = nn.Conv2d(planes, planes, kernel_size=3,
                              padding=1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.do = nn.Dropout2d(dropout)

    def forward(self, x):
        for i in range(self.num_conv):
            x = self.conv[i](x)
            x = self.relu(x)
        x = self.pool(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.do(x)
        return x

class upSampleTransposeConv(nn.Module):
    def __init__(self, inplanes, planes, dropout, upscale_factor=2):
        super(upSampleTransposeConv, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.ConvTranspose2d(inplanes, planes, kernel_size=3,
                              padding=1, stride=2, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(planes)
        self.do = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        self.do(x)
        return x

class Classifier(nn.Module):
    def __init__(self,inplanes,num_classes,poolSize=0,kernelSize=1):
        super(Classifier, self).__init__()
        self.classifier = nn.Conv2d(inplanes,num_classes,kernel_size=kernelSize,padding= kernelSize // 2)
        self.pool = None
        if poolSize > 1:
            self.pool = nn.MaxPool2d(poolSize)

    def forward(self,x):
        if self.pool is not None:
            x = self.pool(x)
        return self.classifier(x)

class FCN(nn.Module):
    def __init__(self, planes, levels, levelDepth, num_classes, kernelSize, dropout):
        super(FCN, self).__init__()

        self.levels = levels
        maxDepth = planes*levels
        self.downLayers = [ConvPool(3,planes,dropout,levelDepth)]
        self.upLayers = [upSampleTransposeConv(maxDepth,maxDepth/2,dropout)]

        for i in range(levels-1):
            self.downLayers.append(ConvPool(pow(2,i)*planes,pow(2,i+1)*planes,dropout,levelDepth))
            self.upLayers.append(upSampleTransposeConv(pow(2,-(i+1))*maxDepth,pow(2,-(i+2))*maxDepth,dropout))



        self.classifier = Classifier(planes,num_classes,kernelSize=kernelSize)

    def forward(self,x):

        inter = [self.downLayers[0](x)]
        for i in range(self.levels-1):
            inter.append( self.downLayers[i](inter[i]) )
        x = self.upLayers[0](inter[self.levels-1])
        for i in range(1, self.levels):
            x = self.upLayers[i](x) + inter[self.levels - 1 - i]

        return self.classifier(x)

