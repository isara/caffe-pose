import caffe
import numpy as np
import sys

caffe_root = '/opt/project/'
input =caffe_root+'models/1miohands-modelzoo-v2/227x227-TRAIN-allImages-forFeatures-0label-227x227handpatch.mean'
output =caffe_root+'models/1miohands-modelzoo-v2/227x227-TRAIN-allImages-forFeatures-0label-227x227handpatch.mean.npy'


blob = caffe.proto.caffe_pb2.BlobProto()
data = open( input , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( output , out )

