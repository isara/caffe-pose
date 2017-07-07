import subprocess
import os
from os.path import basename
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import caffe
import numpy as np

import math

# caffe_root = '/opt/project/caffe-pose/'
caffe_root = '/opt/project/'

folder = 'data/asl'

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
model_def = caffe_root+'models/1miohands-modelzoo-v2/deploy.prototxt'
model_weights = caffe_root+'models/1miohands-modelzoo-v2/1miohands-v2.caffemodel'

# caffe.set_mode_cpu()
caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net (model_def,model_weights,caffe.TEST)

mu = np.load(caffe_root + 'models/1miohands-modelzoo-v2/227x227-TRAIN-allImages-forFeatures-0label-227x227handpatch.mean.npy')
mu = mu.mean(1).mean(1)
print 'mean-subtracted values:', zip('BGR', mu)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

g = open(caffe_root + folder + "/videos.txt")
data = g.readlines()
g.close()

thefile = open(caffe_root + folder + "/handshapes.csv", 'w')


for n, line in enumerate(data, 0):
    name = data[n].replace('\n', '')
    fps = 15

    path, dirs, files = os.walk(caffe_root + folder + '/hands/' + name + '/').next()
    file_count = len(files)

    for i in range(1, file_count):
        frame = i
        input = caffe_root + folder + '/hands/' + name + '/' + str(frame) + '.png'
        if (os.path.exists(input) == True):
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(input))
            pred = net.forward()
            print(name +","+ str(frame) +","+ str(np.argmax(pred['prob'])))
            thefile.writelines(name +","+ str(frame) +","+ str(np.argmax(pred['prob']))+"\n")

thefile.close