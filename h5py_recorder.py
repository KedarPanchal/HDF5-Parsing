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

with open(os.path.join(data_dir, 'bag_dict.json')) as f:

    data = json.load(f)

    with h5py.File(os.path.join(data_dir,'rosbag_h5py.hdf5'), 'w') as file:
        # loop through the bag names
        names = [i['bag_name'] for i in data]
        for name in names:
            # grab bag data
            bag = rosbag.Bag(os.path.join(data_dir, name))
            # create hdf5 group for bag
            group = file.create_group(name)

            for topic, msg, t in bag.read_messages():
                # observations
                # color
                if topic == '/camera/color/image_raw/compressed':
                    try:
                        bridge = CvBridge()

                        cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                        cv_image = cv2.resize(cv_image, (180, 320), interpolation=cv2.INTER_AREA)
                        rgb_arr = np.asarray(cv_image)

                        # cv2.imshow("Color Image", cv_image)
                        # cv2.waitKey(1)


                    except CvBridgeError as e:
                        rospy.logerr("CvBridge Error: {0}".format(e))
                # depth
                elif topic == '/camera/aligned_depth_to_color/image_raw/compressedDepth':
                    pass

                # actions
                # TODO  

            color_dset = group.create_dataset(f"{name}: color images", data=rgb_arr)
            # TODO: depth_dset = group.create_dataset(f"{name}: depth images")
            # TODO: action_dset = group.create_dataset(f"{name}: actions")