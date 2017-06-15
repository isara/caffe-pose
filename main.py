import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.misc import imresize
import caffe
import subprocess
import os
from os.path import basename

# Make sure that caffe is on the python path:
caffe_root = '/root/caffe/project/'
# import sys
# sys.path.insert(0, caffe_root + 'python')

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/heatmap-flic-fusion/matlab.prototxt',
                caffe_root + 'models/heatmap-flic-fusion/caffe-heatmap-flic.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]

folder = 'data/asl'

for filename in os.listdir(caffe_root + folder + "/mp4"):
    if filename.endswith(".mp4"):
        name = os.path.splitext(os.path.basename(filename))[0]
        fps = 30

        cmd1 = "rm %s/square/%s_square.mp4" % (folder, name)
        subprocess.Popen(cmd1, shell=True)

        # Squared and resize the video
        cmd2 = 'ffmpeg -i %s/mp4/%s.mp4 -strict -2 -vf "scale=256:256:force_original_aspect_ratio=decrease,pad=256:256:(ow-iw)/2:(oh-ih)/2" %s/square/%s_square.mp4' % (
            folder, name, folder, name)

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
            # set net to batch size of 50
            net.blobs['data'].reshape(1, 3, 256, 256)
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(
                caffe_root + folder + '/png/' + name + '/' + str(frame) + '.png'))
            net.forward()
            features = net.blobs['conv5_fusion'].data[...][0]

            heatmapResized = np.zeros((7, 256, 256))

            for i in range(0, 7):
                heatmapResized[i] = imresize(features[i], (256, 256), mode='F') - 1

            np.save(caffe_root + 'data/heatmap/' + name + '/' + str(frame) + '.npy', heatmapResized[i])

            # joints = np.zeros((7, 2))
            #
            # for i in range(0, 7):
            #     sub_img = heatmapResized[i]
            #     vec = sub_img.flatten()
            #     idx = np.argmax(vec)
            #
            #     y = (idx.astype('int') / 256)
            #     x = (idx.astype('int') % 256)
            #     joints[i] = np.array([x, y])
            #
            # plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
            #
            # plt.plot([joints[1][0], joints[3][0]], [joints[1][1], joints[3][1]], '.r-', linewidth=3, zorder=1)
            # plt.plot([joints[3][0], joints[5][0]], [joints[3][1], joints[5][1]], '.g-', linewidth=3, zorder=1)
            #
            # plt.plot([joints[2][0], joints[4][0]], [joints[2][1], joints[4][1]], '.r-', linewidth=3, zorder=1)
            # plt.plot([joints[4][0], joints[6][0]], [joints[4][1], joints[6][1]], '.g-', linewidth=3, zorder=1)
            #
            # cmap = cm.get_cmap(name='rainbow')
            # for i in range(0, 7):
            #     plt.scatter(joints[i][0], joints[i][1], color=cmap(i * 256 / 7), s=40, zorder=2)
            #
            # # Convert gif to png
            # cmd1 = "mkdir -p data/output/%s" % (name)
            # subprocess.Popen(cmd1, shell=True)
            #
            # plt.savefig(caffe_root + 'data/output/' + name + '/' + frame + '.png')
            # plt.show()
