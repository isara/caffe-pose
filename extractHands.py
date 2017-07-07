import subprocess
import os
from os.path import basename
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import math

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

# caffe_root = '/opt/project/caffe-pose/'
caffe_root="/Users/rizkyario/Documents/Codes/DeepLearning/caffe-pose/"

folder = 'data/asl'

g = open(caffe_root + folder + "/videos.txt")
data = g.readlines()
g.close()

for n, line in enumerate(data, 0):
    name = data[n].replace('\n', '')
    fps = 15

    path, dirs, files = os.walk(caffe_root + folder + '/png/' + name + '/').next()
    file_count = len(files)

    # dump heatmaps
    cmd1 = "mkdir -p " +folder+"/hands/%s" % (name)
    subprocess.Popen(cmd1, shell=True)

    # frame = 12

    # if (os.path.exists(caffe_root + folder + '/hands/' + name + '/' + str(frame) + '.png')==False):
    #     joints = np.load(caffe_root + folder + '/joints/' + name + '/' + str(frame) + '.npy')
    #     img = Image.open(caffe_root + folder + '/png/' + name + '/' + str(frame) + '.png')
    #     area = (joints[0][0]-25, joints[0][1]-25, joints[0][0]+25, joints[0][1]+25)
    #     cropped_img = img.crop(area)
    #     cropped_img.show()

    for i in range(1, file_count):
        frame = i

        joint = 1
        if(os.path.exists(caffe_root + folder + '/hands/' + name + '/' + str(frame) + '.png')==False):
            if(os.path.exists(caffe_root + folder + '/joints/' + name + '/' + str(frame) + '.npy')==True):
                joints = np.load(caffe_root + folder + '/joints/' + name + '/' + str(frame) + '.npy')

                img = Image.open(caffe_root + folder + '/png/' + name + '/' + str(frame) + '.png')
                area = (joints[joint][1]-50, joints[joint][0]-50, joints[joint][1]+50, joints[joint][0]+50)
                cropped_img = img.crop(area)
                cropped_img.save(caffe_root + folder + '/hands/' + name + '/' + str(frame) + '.png')
            
            