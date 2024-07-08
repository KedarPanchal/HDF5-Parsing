import json
import sys
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
from tf import ExtrapolationException

from numpy.linalg import inv

import h5py

from tf.transformations import quaternion_matrix
from tf.transformations import quaternion_from_matrix

import tf_bag
import tf

rospy.set_param('use_sim_time', 'true')


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
        i = 0

        for topic, msg, t in bag.read_messages():
            i+=1 
            if i == 1: 
                t_first_tf = t

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
            # gripper state
            elif topic == '/bariflex':
                pass
            # transforms
            elif topic == '/tf':
                # tf lookup
                try: 
                    trans, quat = bag_transformer.lookupTransform('camera_link', 'map', t)
                    # idk how to compute 2 quaternions so just going to compute the pose and extract translation and quaternion!!
                    '''
                    # Quaternion subtraction is different from regular, element-wise subtraction
                    cq_x, cq_y, cq_z, cq_w = quat
                    pq_x, pq_y, pq_z, pq_w = prev_quat

                    # Compute the conjugate of the previous orientation
                    pq_conj_w = pq_w
                    pq_conj_x = -pq_x
                    pq_conj_y = -pq_y
                    pq_conj_z = -pq_z

                    # Quaternion multiplication (prev_conjugate * current_orientation)
                    dq_w = pq_conj_w * cq_w - pq_conj_x * cq_x - pq_conj_y * cq_y - pq_conj_z * cq_z
                    dq_x = pq_conj_w * cq_x + pq_conj_x * cq_w + pq_conj_y * cq_z - pq_conj_z * cq_y
                    dq_y = pq_conj_w * cq_y - pq_conj_x * cq_z + pq_conj_y * cq_w + pq_conj_z * cq_x
                    dq_z = pq_conj_w * cq_z + pq_conj_x * cq_y - pq_conj_y * cq_x + pq_conj_z * cq_w

                    ct_x, ct_y, ct_z = trans
                    pt_x, pt_y, pt_z = prev_trans

                    dt_x = ct_x - pt_x
                    dt_y = ct_y - pt_y
                    dt_z = ct_z - pt_z

                    prev_quat = quat
                    prev_trans = trans
                    
                    uber_action_arr.append([dt_x, dt_y, dt_z, dq_x, dq_y, dq_z, dq_w, baridata])
                    '''
                    if(colorDeque and depthDeque):
                        # uber_rgb_arr.append[colorDeque[-1*colorOffset]]
                        # uber_depth_arr.append[depthDeque[-1*depthOffset]]
                        # colorOffset = 0
                        # depthOffset = 0
                        pass
                    
                except ExtrapolationException as e:
                    print(type(e))
                    continue
                    # print(f"Exception {e}", file=sys.stderr)
          
                    





        # creates the datasets
        print(np.array(uber_rgb_arr).shape)
        print(np.array(uber_depth_arr).shape)
        print(np.array(uber_action_arr).shape)
        color_dset = group.create_dataset(f"{name}: color images", data=np.array(uber_rgb_arr))
        depth_dset = group.create_dataset(f"{name}: depth images", data=np.array(uber_depth_arr))
        action_dset = group.create_dataset(f"{name}: actions", data=np.array(uber_action_arr))