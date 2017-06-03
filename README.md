#### Make sure XQuartz is running
open -a XQuartz
xhost +

Details: http://riz.ky/running-gui-docker/

#### Caffe-heatmap

docker build -t rizkyario/caffe .
docker run -it --name caffe  \
-v /Users/rizkyario/Documents/Codes/DeepLearning/caffe-pose:/root/caffe/project rizkyario/caffe /bin/bash

added some needed library such ass
- python-tk 
- ffmpeg

