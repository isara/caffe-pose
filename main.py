import caffe
import numpy as np

caffe_root = '/opt/project/'

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
model_def = caffe_root+'models/1miohands-modelzoo-v2/deploy.prototxt'
model_weights = caffe_root+'models/1miohands-modelzoo-v2/1miohands-v2.caffemodel'

caffe.set_mode_cpu()
net = caffe.Net (model_def,model_weights,caffe.TEST)

file = []
file+=["data/1miohands/images/final_phoenix_noPause_noCompound_lefthandtag_noClean/28November_2011_Monday_tagesschau_default-12/1/*.png_fn000178-0.png"]
file+=["data/1miohands/images/asl/1.png"]
file+=["data/1miohands/images/asl/4.png"]
file+=["data/asl/hands/zero/24.png"]


mu = np.load(caffe_root + 'models/1miohands-modelzoo-v2/227x227-TRAIN-allImages-forFeatures-0label-227x227handpatch.mean.npy')
mu = mu.mean(1).mean(1)
print 'mean-subtracted values:', zip('BGR', mu)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


for f in file:
    net.blobs['data'].data[...] = transformer.preprocess('data',caffe.io.load_image(caffe_root+f))
    pred =net.forward()
    print(np.argmax( pred['prob']))
