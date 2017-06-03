import subprocess

root = '/root/caffe/project/'

folder = 'data/asl'
name = 'age'
fps = 15

cmd1 = "%s/mp4/%s_square.mp" % (folder, name)
subprocess.Popen(cmd1, shell=True)

# Squared and resize the video
cmd2 = 'ffmpeg -i %s/mp4/%s.mp4 -strict -2 -vf "scale=256:256:force_original_aspect_ratio=decrease,pad=256:256:(ow-iw)/2:(oh-ih)/2" %s/mp4/%s_square.mp4' % (folder, name, folder, name)

# Convert gif to png

cmd3 = "ffmpeg -i %s/mp4/%s_square.mp4 -vcodec png -r %d %s/png/%s/%%0d.png" % (folder, name, fps, folder, name)

cmd1 = "mkdir -p %s/png/%s/" % (folder, name)
subprocess.Popen(cmd1, shell=True)

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
