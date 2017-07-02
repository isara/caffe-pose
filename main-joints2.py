import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.misc import imresize
import caffe
import subprocess
import os
from os.path import basename
import boto
import boto.s3.connection
from boto.s3.key import Key

def define_center_one_frame_one_joint(joint_image, size_sliding_window = 10):
    '''
    define_center_one_frame_one_join defines for one frame and one joint heatmap the argmax of the joint position (using a sliding window avoids defining a isolated point)
    joint_image : Array(n,m) - an array that represents the heatmap for a particular joint
    size_sliding_window : Integer - the size of sliding window to compute the sum on
    '''
    joint_image_sum = np.zeros((joint_image.shape[0]-size_sliding_window, joint_image.shape[1]-size_sliding_window))
    for i in range(joint_image.shape[0]-size_sliding_window):
        for j in range(joint_image.shape[1]-size_sliding_window):
            joint_image_sum[i,j] = np.sum(joint_image[i:i+size_sliding_window,j:j+size_sliding_window])
    return np.unravel_index(np.argmax(joint_image_sum), joint_image_sum.shape)

def define_center_one_frame_every_joints(one_frame_image, name, frame, size_sliding_window = 10):
    '''
    define_center_one_frame_every_joins defines for every joints its argmax relative to the heatmap
    one_frame_image : Array(nb_joints,n,m) - an array that represents the heatmap for every joints
    size_sliding_window : Integer - the size of sliding window to compute the sum on
    '''
    list_joints_location = []
    for i in range(one_frame_image.shape[0]):
        list_joints_location.append(define_center_one_frame_one_joint(one_frame_image[i], size_sliding_window))

    np.save(caffe_root + 'data/joints/' + name + '/' + str(frame) + '.npy', list_joints_location)

caffe_root = '/opt/project/'
# caffe_root="/Users/rizkyario/Documents/Codes/DeepLearning/caffe-pose/"

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe.set_mode_cpu()
# caffe.set_mode_gpu()
# caffe.set_device(0)
net = caffe.Net(caffe_root + 'models/heatmap-flic-fusion/matlab.prototxt',
                caffe_root + 'models/heatmap-flic-fusion/caffe-heatmap-flic.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]

folder = 'data/asl'

g = open(caffe_root + folder + "/videos.txt")
data = g.readlines()
g.close()

for n, line in enumerate(data, 0):
    name = data[n].replace('\n', '')
    fps = 15

    cmd1 = "rm %s/square/%s_square.mp4" % (folder, name)
    subprocess.Popen(cmd1, shell=True)

    # Squared and resize the video
    cmd2 = 'ffmpeg -i %s/mp4/%s.mp4 -strict -2 -vf "scale=256:256:force_original_aspect_ratio=decrease,pad=256:256:(ow-iw)/2:(oh-ih)/2" %s/square/%s_square.mp4' % (folder, name, folder, name)
    subprocess.Popen(cmd1, shell=True)

    # Convert gif to png
    cmd1 = "mkdir -p %s/png/%s/" % (folder, name)
    subprocess.Popen(cmd1, shell=True)

    cmd3 = "ffmpeg -i %s/square/%s_square.mp4 -vcodec png -r %d %s/png/%s/%%0d.png" % (folder, name, fps, folder, name)

    x = subprocess.Popen(cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = x.communicate()
    if output:
        print "success ", output
    else:
        print "error ", err

    x = subprocess.Popen(cmd3, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = x.communicate()
    if output:
        print "success ", output
    else:
        print "error ", err

    path, dirs, files = os.walk(caffe_root + folder + '/png/' + name + '/').next()
    file_count = len(files)

    # dump heatmaps
    cmd1 = "mkdir -p data/joints/%s" % (name)
    subprocess.Popen(cmd1, shell=True)

    for i in range(1, file_count):
        frame = i

        if(os.path.exists(caffe_root + 'data/joints/' + name + '/' + str(frame) + '.npy')==False):
            # set net to batch size of 50
            net.blobs['data'].reshape(1, 3, 256, 256)
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(
                caffe_root + folder + '/png/' + name + '/' + str(frame) + '.png'))
            net.forward()
            features = net.blobs['conv5_fusion'].data[...][0]

            heatmapResized = np.zeros((7, 256, 256))

            for i in range(0, 7):
                heatmapResized[i] = imresize(features[i], (256, 256), mode='F') - 1

            define_center_one_frame_every_joints(heatmapResized, name, frame, 10)

