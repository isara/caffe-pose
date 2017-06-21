import os 
import boto
import boto.s3.connection
from boto.s3.key import Key

caffe_root="/Users/rizkyario/Documents/Codes/DeepLearning/caffe-pose/"
name = "1_dollar"
frame = "1"
try:

    conn = boto.s3.connect_to_region('us-east-1',
    aws_access_key_id = 'AKIAIYMOQ6GQZ7H5H7PA',
    aws_secret_access_key = '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',
    calling_format = boto.s3.connection.OrdinaryCallingFormat(),
    )

    bucket = conn.get_bucket('isara')
    file = caffe_root + 'data/heatmap/' + name + '/' + str(frame) + '.npy'
    path = 'heatmap/'+name #Directory Under which file should get upload
    full_key_name = os.path.join(path, name+ '.npy')
    k = bucket.new_key(full_key_name)
    k.set_contents_from_filename(file)
    path = caffe_root + 'data/heatmap/' + name + '/' + str(frame) + '.npy'

    os.remove(path)

except Exception,e:
    print str(e)
    print "error"   