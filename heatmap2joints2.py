# from eventlet import *
# patcher.monkey_patch(all=True)
import boto
import os, sys, time
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket

import numpy as np


def define_center_one_frame_one_joint(joint_image, size_sliding_window=10):
    '''
    define_center_one_frame_one_join defines for one frame and one joint heatmap the argmax of the joint position (using a sliding window avoids defining a isolated point)
    joint_image : Array(n,m) - an array that represents the heatmap for a particular joint
    size_sliding_window : Integer - the size of sliding window to compute the sum on
    '''
    joint_image_sum = np.zeros((joint_image.shape[0] - size_sliding_window, joint_image.shape[1] - size_sliding_window))
    for i in range(joint_image.shape[0] - size_sliding_window):
        for j in range(joint_image.shape[1] - size_sliding_window):
            joint_image_sum[i, j] = np.sum(joint_image[i:i + size_sliding_window, j:j + size_sliding_window])
    return np.unravel_index(np.argmax(joint_image_sum), joint_image_sum.shape)


def define_center_one_frame_every_joints(one_frame_image, size_sliding_window=10):
    '''
    define_center_one_frame_every_joins defines for every joints its argmax relative to the heatmap
    one_frame_image : Array(nb_joints,n,m) - an array that represents the heatmap for every joints
    size_sliding_window : Integer - the size of sliding window to compute the sum on
    '''
    list_joints_location = []
    for i in range(one_frame_image.shape[0]):
        list_joints_location.append(define_center_one_frame_one_joint(one_frame_image[i], size_sliding_window))
    return list_joints_location


def heatmaps2joins(folder, file):
    heatmap_dir = '/Users/rizkyario/Documents/Codes/DeepLearning/caffe-pose/data/heatmap/'
    joints_dir = '/Users/rizkyario/Documents/Codes/DeepLearning/caffe-pose/data/joints/'

    if not os.path.exists(heatmap_dir + folder):
        os.makedirs(heatmap_dir + folder)
        os.makedirs(joints_dir + folder)

    heatmap_file = '/Users/rizkyario/Documents/Codes/DeepLearning/caffe-pose/data/' + key.name
    res = key.get_contents_to_filename(heatmap_file)

    heatmapResized = np.load(heatmap_file)
    list_joints_location = define_center_one_frame_every_joints(heatmapResized, 10)
    np.save(joints_dir + folder + file, list_joints_location)
    os.remove(heatmap_file)


def download_file(key_name):
    # Its imp to download the key from a new connection
    conn = boto.s3.connect_to_region('us-east-1',
                                     aws_access_key_id='AKIAIYMOQ6GQZ7H5H7PA',
                                     aws_secret_access_key='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',
                                     calling_format=boto.s3.connection.OrdinaryCallingFormat(),
                                     )
    bucket = Bucket(connection=conn, name="isara")

    key = bucket.get_key(key_name)

    res = key.get_contents_to_filename(key.name)

if __name__ == "__main__":
    conn = boto.s3.connect_to_region('us-east-1',
                                     aws_access_key_id='AKIAIYMOQ6GQZ7H5H7PA',
                                     aws_secret_access_key='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',
                                     calling_format=boto.s3.connection.OrdinaryCallingFormat(),
                                     )
    bucket = Bucket(connection=conn, name="isara")

    # logging.info("Fetching bucket list")
    bucket_list = bucket.list(prefix="PREFIX")

    # logging.info("Creating a pool")
    pool = GreenPool(size=20)

    # logging.info("Saving files in bucket...")
    for key in bucket.list():
        print key.name
        if 'heatmap/' in key.name:
            folder = key.name.split('/')[1] + '/'
            file = key.name.split('/')[2]
            pool.spawn_n(heatmaps2joins, folder, file)
    pool.waitall()