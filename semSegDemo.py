import torch
from torch.utils import data
from model import  FCN
from dataset import SSDataSet
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, ToPILImage
from PIL import Image
import sys
import progressbar
import numpy as np
import cv2

def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.float)
    cmap[0, 0] = 0
    cmap[0, 1] = 0
    cmap[0, 2] = 0
    cmap[1, 0] = 1
    cmap[1, 1] = 0
    cmap[1, 2] = 0
    cmap[2, 0] = 0
    cmap[2, 1] = 1
    cmap[2, 2] = 0
    cmap[3, 0] = 0
    cmap[3, 1] = 0
    cmap[3, 2] = 1
    cmap[4, 0] = 1
    cmap[4, 1] = 1
    cmap[4, 2] = 1
    cmap[5, 0] = 0
    cmap[5, 1] = 1
    cmap[5, 2] = 1
    cmap[6, 0] = 1
    cmap[6, 1] = 0
    cmap[6, 2] = 1
    cmap[7, 0] = 1
    cmap[7, 1] = 1
    cmap[7, 2] = 0
    return cmap

def Colorize(gray_image,n=8):
        cmap = labelcolormap(n)
        cmap = torch.from_numpy(cmap[:n]).float()
        size = gray_image.size()
        color_image = torch.FloatTensor(3, size[0], size[1]).fill_(0)

        for label in range(0, len(cmap)):
            mask = (label == gray_image).cpu()
            color_image[0][mask] = cmap[label][0]
            color_image[1][mask] = cmap[label][1]
            color_image[2][mask] = cmap[label][2]

        return color_image


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

    trBack = Compose([
        Normalize([-.5, -.5, -.5], [1, 1, 1]),
    ])

    root = 'C:/data/cityscapes/' if sys.platform == 'win32' else '~/data/cityscapes'

    #sampler = torch.utils.data.sampler.SubsetRandomSampler(range(64))

    valloader = data.DataLoader(SSDataSet(root, split="val", img_transform=input_transform,
                                             label_transform=target_transform), #sampler=sampler,
                                  batch_size=1, shuffle=False, num_workers=4)


    numClass = 8
    numPlanes = 16
    levels = 4
    levelDepth = 2
    kernelSize = 3

    model = FCN(numPlanes,levels,levelDepth,numClass,kernelSize,0.1)

    mapLoc = None if haveCuda else {'cuda:0': 'cpu'}

    if haveCuda:
        model = model.cuda()

    model.load_state_dict(torch.load(root + 'bestModelSeg.pth',map_location=mapLoc))

    model.eval()
    for i, (images, labels) in enumerate(valloader):
        if torch.cuda.is_available():
            images = images.cuda()

        pred = model(images)

        _, predClass = torch.max(pred, 1)

        #img = Image.fromarray(Colorize(predClass[0]).permute(1, 2, 0).numpy().astype('uint8'))
        orig = trBack(images[0].cpu()).numpy()
        img = Colorize(predClass[0]).numpy()
        img = (0.5*img+0.5*orig).transpose(1,2,0)
        img = cv2.resize(img,dsize=None,fx=4,fy=4)
        print(img.shape)
        cv2.imshow('Image',img)
        cv2.waitKey(1000)
