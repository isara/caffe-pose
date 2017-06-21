import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.misc import imresize
import subprocess
import os
from os.path import basename

import caffe

caffe_root = '/root/caffe/project/'

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]


heatmapResized = np.load('/Users/rizkyario/Documents/Codes/DeepLearning/caffe-pose/data/heatmap2/excuse/1.npy')

joints = np.zeros((7, 2))

for i in range(0, 7):
    sub_img = heatmapResized[i]
    vec = sub_img.flatten()
    idx = np.argmax(vec)

    y = (idx.astype('int') / 256)
    x = (idx.astype('int') % 256)
    joints[i] = np.array([x, y])

plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))

plt.plot([joints[1][0], joints[3][0]], [joints[1][1], joints[3][1]], '.r-', linewidth=3, zorder=1)
plt.plot([joints[3][0], joints[5][0]], [joints[3][1], joints[5][1]], '.g-', linewidth=3, zorder=1)

plt.plot([joints[2][0], joints[4][0]], [joints[2][1], joints[4][1]], '.r-', linewidth=3, zorder=1)
plt.plot([joints[4][0], joints[6][0]], [joints[4][1], joints[6][1]], '.g-', linewidth=3, zorder=1)

cmap = cm.get_cmap(name='rainbow')
for i in range(0, 7):
    plt.scatter(joints[i][0], joints[i][1], color=cmap(i * 256 / 7), s=40, zorder=2)

# Convert gif to png
cmd1 = "mkdir -p data/output/%s" % (name)
subprocess.Popen(cmd1, shell=True)

plt.savefig(caffe_root + 'data/output/' + name + '/' + frame + '.png')
plt.show()
