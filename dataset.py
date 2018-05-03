import torch
import os.path as osp
from torch.utils import data
import glob
from PIL import Image
import numpy as np
import random
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def toLabel(x):
    return torch.squeeze(x.long())


class SSDataSet(data.Dataset):
    def __init__(self, root, split="train", img_transform=None, label_transform=None):
        self.root = root
        self.split = split
        self.images = []
        self.labels = []
        self.img_transform = img_transform
        self.label_transform = label_transform

        img_dir = osp.join(root, "images")
        lab_dir = osp.join(root, "labels")
        self.img_dir = osp.join(img_dir,split)
        self.lab_dir = osp.join(lab_dir,split)

        for file in sorted(glob.glob1(self.img_dir, "*.png"),key=alphanum_key):
            self.images.append(file)
        for file in sorted(glob.glob1(self.lab_dir, "*.png"),key=alphanum_key):
            self.labels.append(file)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_file = osp.join( self.img_dir, self.images[index])
        lab_file = osp.join( self.lab_dir, self.labels[index])

        img = Image.open(img_file).convert('RGB')
        label = Image.open(lab_file).convert("I")

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        if self.img_transform is not None:
            imgs = self.img_transform(img)
        else:
            imgs = img

        random.seed(seed)  # apply this seed to target tranfsorms
        if self.label_transform is not None:
            labels = self.label_transform(label)
        else:
            labels = label

        return imgs, toLabel(labels)
