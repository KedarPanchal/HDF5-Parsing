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
import tf_bag




script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(script_dir, 'data')
saved_mapping_dir = os.path.join(data_dir, "saved_mapping_bags")

with open(os.path.join(data_dir, 'bag_dict.json')) as f, h5py.File(os.path.join(data_dir,'rosbag_h5py.hdf5'), 'w') as file:

    data = json.load(f)

    # creates the file in the data folder in "write" mode
    # loop through the bag names
    names = [i['bag_name'] for i in data]
    for name in names:
        # grab bag data
        try:
            bag = rosbag.Bag(os.path.join(data_dir, name))
        except FileNotFoundError:
            bag = rosbag.Bag(os.path.join(saved_mapping_dir, name))

        # create hdf5 group for bag
        group = file.create_group(name)

        uber_rgb_arr = []
        uber_depth_arr = []
        header_color, header_depth = None, None
        t_prev = 0

        for topic, msg, t in bag.read_messages():
            # observations
            # color
            if topic == '/camera/color/image_raw/compressed':
                try:
                    bridge = CvBridge()
                    cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                    cv_image = cv2.resize(cv_image, (320, 180), interpolation=cv2.INTER_AREA)
                    rgb_arr = np.asarray(cv_image)
                    uber_rgb_arr.append(rgb_arr)
                    cv2.imshow("Color Image", cv_image)
                    cv2.waitKey(1)


                except CvBridgeError as e:
                    rospy.logerr("CvBridge Error: {0}".format(e))
            # depth
            elif topic == '/camera/aligned_depth_to_color/image_raw/compressedDepth':
                data = np.frombuffer(msg.data[12:], np.uint8)
                cv_image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                cv_image = cv2.resize(cv_image, (320, 180), interpolation=cv2.INTER_AREA)
                depth_arr = np.asarray(cv_image) # could also just be data but check to be sure
                uber_depth_arr.append(depth_arr)
                cv2.imshow("Depth Image", cv_image)
                cv2.waitKey(1)
                
            # actions
            bag_transformer = tf_bag.BagTfTransformer(bag)

        # creates the datasets
        print(np.array(uber_rgb_arr).shape)
        print(np.array(uber_depth_arr).shape)
        color_dset = group.create_dataset(f"{name}: color images", data=np.array(uber_rgb_arr))
        depth_dset = group.create_dataset(f"{name}: depth images", data=np.array(uber_depth_arr))
        # TODO: action_dset = group.create_dataset(f"{name}: actions")