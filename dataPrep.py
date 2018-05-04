import os
import glob
from shutil import copyfile
from PIL import Image
from torchvision.transforms import Resize

imgRes = Resize(128)
labRes = Resize(128,interpolation=Image.NEAREST)

def copyLabels(root,split,dest):

    srcPath = os.path.join(root,split)
    dstPath = os.path.join(dest,split)

    print(srcPath)

    for file in glob.glob(srcPath+"/**/*.png"):
        dst = os.path.join(dstPath,file.split('/')[-1])
        copyfile(file,dst)

def resize(root):
    imgpath = os.path.join(root,"images")
    labpath = os.path.join(root,"labels")

    for file in glob.glob(imgpath+"/**/*.png"):
        I = Image.open(file).convert('RGB')
        I = imgRes(I)
        I.save(file)

    for file in glob.glob(labpath+"/**/*.png"):
        I = Image.open(file).convert("I")
        I = labRes(I)
        I.save(file)


'''root = '/Users/martonszemenyei/Downloads/cityscapes/leftImg8bit'
dest = '/Users/martonszemenyei/Downloads/cityscapes/images'
copyLabels(root,'train',dest)
copyLabels(root,'val',dest)
copyLabels(root,'test',dest)'''

root = 'E:/cityscapes'
resize(root)