import json
import os
import rosbag
import numpy as np
import rospy
from sensor_msgs.msg import Image, CompressedImage
import cv2
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg
import argparse

from collections import deque

from numpy.linalg import inv

import h5py

from tf.transformations import quaternion_matrix
from tf.transformations import euler_from_matrix
script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(script_dir, 'data')


with open('/home/kpanchal/Documents/HSRA/Summer-Camp-Rosbag/data/bag_dict.json') as f:
    f = h5py.File('rosbag_h5py.hdf5', 'w')

    data = json.load(f)

    color_images = []

    names = [i['bag_name'] for i in data]
    for name in names:
        
        bag = rosbag.Bag(os.path.join(data_dir, name))

        group = f.create_group(name)
        color_dset = f.create_dataset("color images",)
        depth_dset = f.create_dataset("depth images",)

        for topic, msg, t in bag.read_messages():
            if topic == '/camera/color/image_raw/compressed':
                try:
                    bridge = CvBridge()

                    cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                    cv_image = cv2.resize(cv_image, (180, 320), interpolation=cv2.INTER_AREA)
                    rgb_arr = np.asarray(cv_image)
                    print(rgb_arr.shape)
                    # cv2.imshow("Color Image", cv_image)
                    # cv2.waitKey(1)


                except CvBridgeError as e:
                    rospy.logerr("CvBridge Error: {0}".format(e))

            elif topic == '/camera/aligned_depth_to_color/image_raw/compressedDepth':
                pass
            


    