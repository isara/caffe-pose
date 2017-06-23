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


caffe_root = '/root/caffe/project/'
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
    cmd1 = "mkdir -p data/heatmap/%s" % (name)
    subprocess.Popen(cmd1, shell=True)

    for i in range(1, file_count):
        frame = i

        if(os.path.exists(caffe_root + 'data/heatmap/' + name + '/' + str(frame) + '.npy')==False):
            # set net to batch size of 50
            net.blobs['data'].reshape(1, 3, 256, 256)
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(
                caffe_root + folder + '/png/' + name + '/' + str(frame) + '.png'))
            net.forward()
            features = net.blobs['conv5_fusion'].data[...][0]

            heatmapResized = np.zeros((7, 256, 256))

            for i in range(0, 7):
                heatmapResized[i] = imresize(features[i], (256, 256), mode='F') - 1

            np.save(caffe_root + 'data/heatmap/' + name + '/' + str(frame) + '.npy', heatmapResized)

            try:
                conn = boto.s3.connect_to_region('us-east-1',
                                                 aws_access_key_id='AKIAIYMOQ6GQZ7H5H7PA',
                                                 aws_secret_access_key='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',
                                                 calling_format=boto.s3.connection.OrdinaryCallingFormat(),
                                                 )

                bucket = conn.get_bucket('isara')
                file = caffe_root + 'data/heatmap_2/' + name + '/' + str(frame) + '.npy'
                path = 'heatmap/' + name  # Directory Under which file should get upload
                full_key_name = os.path.join(path, name + '.npy')
                k = bucket.new_key(full_key_name)
                k.set_contents_from_filename(file)
                path = caffe_root + 'data/heatmap/' + name + '/' + str(frame) + '.npy'

                os.remove(path)

            except Exception, e:
                print str(e)
                print "error"

