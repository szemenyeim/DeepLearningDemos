import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Pixelwise Cross-entropy loss
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs,dim=1), targets)


class ConvPool(nn.Module):
    def __init__(self, inplanes, planes, dropout, num_conv=1):
        super(ConvPool, self).__init__()
        self.relu = nn.ReLU()
        self.num_conv = num_conv
        self.conv = nn.Sequential()
        self.conv.add_module("InConv",nn.Conv2d(inplanes, planes, kernel_size=3, dilation=2,
                              padding=2, bias=False))
        for i in range(self.num_conv-1):
            self.conv.add_module( ( "Conv%d" % (i+1) ) , nn.Conv2d(planes, planes, kernel_size=3, dilation=2,
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
        self.conv = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, padding=1, stride=2, output_padding=1, bias=True)
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
        maxDepth = pow(2,levels-1)*planes
        self.downLayers = nn.Sequential()

        self.downLayers.add_module("InConv",nn.Conv2d(3,planes,kernelSize,padding=kernelSize//2))
        #self.downLayers = [nn.Conv2d(3,planes,kernelSize,padding=kernelSize//2)]
        self.upLayers = nn.Sequential()#[]

        for i in range(levels):
            self.downLayers.add_module( ("Down%d" % (i+1) ),ConvPool(int(pow(2,i)*planes),int(pow(2,i+1)*planes),dropout,levelDepth))
            self.upLayers.add_module( ("Up%d" % i ),upSampleTransposeConv(int(pow(2,-(i-1))*maxDepth),int(pow(2,-(i))*maxDepth),dropout))


        self.classifier = Classifier(planes,num_classes,kernelSize=kernelSize)

    def forward(self,x):

        inter = [self.downLayers[0](x)]
        for i in range(self.levels):
            inter.append( self.downLayers[i+1](inter[i]) )
        x = self.upLayers[0](inter[self.levels])
        for i in range(1, self.levels):
            x = self.upLayers[i](x) + inter[self.levels - 1 - i]

        return self.classifier(x)

