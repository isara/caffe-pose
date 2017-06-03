import subprocess

subprocess.Popen("pwd", shell=True)

cmd1 = "sh train_heatmap.sh heatmap-flic-fusion-1"
x = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output, err = x.communicate()
if output:
    print "success ", output
else:
    print "error ", err