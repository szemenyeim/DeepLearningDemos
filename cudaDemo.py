import torch
from torch import nn
from torch.autograd import Variable

useCuda = True if torch.cuda.is_available() else False

class myModel(nn.Module):
    def __init__(self):
        super(myModel,self).__init__()
        self.conv = nn.Conv2d(3,2,3)

    def forward(self, x):
        return self.conv(x)

myNet = myModel()
myLoss = nn.CrossEntropyLoss()
myData = torch.FloatTensor(2,3,3,3)

# ALIAS: cuda esetén cuda tensor lesz, egyébként meg cpu
myLongTensor = torch.cuda.LongTensor if useCuda else torch.LongTensor

result = myLongTensor(2)
result[1] = result[0] = 0

if useCuda:
    myNet = myNet.cuda()
    myData = myData.cuda()
    myLoss = myLoss.cuda()

result = Variable(result)
myData = Variable(myData)

out = torch.squeeze(myNet(myData))
loss = myLoss(out,result)
print(loss)



