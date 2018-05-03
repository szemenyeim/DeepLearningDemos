import os
import glob
from shutil import copyfile

def copyLabels(root,split,dest):

    srcPath = os.path.join(root,split)
    dstPath = os.path.join(dest,split)

    print srcPath

    for file in glob.glob(srcPath+"/**/*.png"):
        dst = os.path.join(dstPath,file.split('/')[-1])
        copyfile(file,dst)



root = '/Users/martonszemenyei/Downloads/cityscapes/leftImg8bit'
dest = '/Users/martonszemenyei/Downloads/cityscapes/images'
copyLabels(root,'train',dest)
copyLabels(root,'val',dest)
copyLabels(root,'test',dest)