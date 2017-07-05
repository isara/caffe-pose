import subprocess
import os
from os.path import basename

# caffe_root = '/opt/project/'
caffe_root="/Users/rizkyario/Documents/Codes/DeepLearning/caffe-pose/"


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