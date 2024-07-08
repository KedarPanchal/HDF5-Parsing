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
import time
from collections import deque


from numpy.linalg import inv

import h5py

from tf.transformations import quaternion_matrix
from tf.transformations import quaternion_from_matrix

import tf_bag
import tf



script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(script_dir, 'data')

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
            continue

        # create hdf5 group for bag
        group = file.create_group(name)

        uber_rgb_arr = []
        uber_depth_arr = []
        uber_action_arr = []
        prev_pose = None
        prev_trans = None
        prev_quat = None
        test = 0

        colorDeque = deque()
        depthDeque = deque()

        colorOffset = 0
        depthOffset = 0

        bag_transformer = tf_bag.BagTfTransformer(bag)

        trans = tf.Transformer(True, rospy.Duration(10.0))


        # print(f"Sum: {sum(1 for _ in bag.read_messages())}")
        t_prev = 0

        for topic, msg, t in bag.read_messages():
            # observations
            # color
            if topic == '/camera/color/image_raw/compressed':
                # print(f"color: {t}")
                try:
                    bridge = CvBridge()
                    cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                    cv_image = cv2.resize(cv_image, (320, 180), interpolation=cv2.INTER_AREA)
                    rgb_arr = np.asarray(cv_image)

                    colorDeque.append(rgb_arr)
                    colorOffset += 1

                    # cv2.imshow("Color Image", cv_image)
                    # cv2.waitKey(1)
                except CvBridgeError as e:
                    rospy.logerr("CvBridge Error: {0}".format(e))
            # depth
            elif topic == '/camera/aligned_depth_to_color/image_raw/compressed':
                # print(f"depth: {t}")
                try:
                    # print(msg.format)
                    data = np.frombuffer(msg.data, np.uint8)
                    depth_image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                    depth_image = cv2.resize(depth_image, (320, 180), interpolation=cv2.INTER_AREA)
                    depth_arr = np.asarray(depth_image)

                    depthDeque.append(depth_arr)
                    depthOffset += 1

                    cv2.imshow("Depth Image", depth_image)
                    cv2.waitKey(1)
                except CvBridgeError as e:
                    rospy.logerr("CvBridge Error: {0}".format(e))
                # try:
                #     bridge = CvBridge()
                #     cv_image = bridge.compressed_imgmsg_to_cv2(msg, "16UC1")
                #     cv_image = cv2.resize(cv_image, (320, 180), interpolation=cv2.INTER_AREA)
                #     depth_arr = np.asarray(cv_image)
                #     uber_depth_arr.append(depth_arr)
                #     # cv2.imshow("Color Image", cv_image)
                #     # cv2.waitKey(1)
                # except CvBridgeError as e:
                #     rospy.logerr("CvBridge Error: {0}".format(e))

                # data = np.frombuffer(msg.data[12:], np.uint8)to
                # cv_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                # cv_image = cv2.resize(cv_image, (320, 180), interpolation=cv2.INTER_AREA)
                # depth_arr = np.asarray(cv_image) # could also just be data but check to be sure
                # uber_depth_arr.append(depth_arr)
                # cv2.imshow("Depth Image", cv_image)
                # cv2.waitKey(1)
                # pass
                
            # actions
            elif topic == '/tf':
                # tf lookup
                if msg.transforms[0].header.frame_id == "odom":
                    test = test + 1
                    print(f"tf: {test}")

                # trans, quat = bag_transformer.lookupTransform('camera_link', 'map', t)
                # idk how to compute 2 quaternions so just going to compute the pose and extract translation and quaternion!!
                '''                
                # Quaternion subtraction is different from regular, element-wise subtraction
                curr_w, curr_x, curr_y, curr_z = quaternion
                prev_w, prev_x, prev_y, prev_z = prev_quat

                # Compute the conjugate of the previous orientation
                prev_conj_w = prev_w
                prev_conj_x = -prev_x
                prev_conj_y = -prev_y
                prev_conj_z = -prev_z

                # Quaternion multiplication (prev_conjugate * current_orientation)
                delta_w = prev_conj_w * curr_w - prev_conj_x * curr_x - prev_conj_y * curr_y - prev_conj_z * curr_z
                delta_x = prev_conj_w * curr_x + prev_conj_x * curr_w + prev_conj_y * curr_z - prev_conj_z * curr_y
                delta_y = prev_conj_w * curr_y - prev_conj_x * curr_z + prev_conj_y * curr_w + prev_conj_z * curr_x
                delta_z = prev_conj_w * curr_z + prev_conj_x * curr_y - prev_conj_y * curr_x + prev_conj_z * curr_w


                mat = quaternion_matrix(quaternion)
                transform_matrix = np.identity(4)
                transform_matrix[:3, :3] = mat[:3, :3]
                transform_matrix[:3, 3] = translation[:3]

                if prev_pose is not None:
                    inv_prev = inv(prev_pose)
                    rel_pose = np.matmul(inv_prev, transform_matrix)
                else:
                    rel_pose = transform_matrix

                translation[:3] = rel_pose[:3, 3]
                quaternion = quaternion_from_matrix(rel_pose[:3, :3])
                prev_pose = transform_matrix
                
                # NOTE: saving just the translation and quaternion, can do the calcs using these later
                uber_action_arr.append((translation, quaternion))
                '''
                if(colorDeque and depthDeque):
                    uber_rgb_arr.append[-colorDeque]
                    uber_depth_arr.append[-depthDeque]
                    colorDeque = 0
                    depthDeque = 0
                    





        # creates the datasets
        print(np.array(uber_rgb_arr).shape)
        print(np.array(uber_depth_arr).shape)
        print(np.array(uber_action_arr).shape)
        color_dset = group.create_dataset(f"{name}: color images", data=np.array(uber_rgb_arr))
        depth_dset = group.create_dataset(f"{name}: depth images", data=np.array(uber_depth_arr))
        action_dset = group.create_dataset(f"{name}: actions", data=np.array(uber_action_arr))