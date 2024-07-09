import json
import re
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

import h5py_helper as h5h

script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(script_dir, 'data')

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

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
            print(f"{name} not found, continuing")
            continue

        # create hdf5 group for bag
        group = file.create_group(name)

        uber_rgb_arr = []
        uber_depth_arr = []
        uber_action_arr = []
        uber_bari_arr = []

        prev_pose = None
        deltaTrans = [0, 0, 0]
        deltaQuat = [0, 0, 0, 1]

        colorList = []
        depthList = []
        bariList = []

        colorOffset = 0
        depthOffset = 0
        bariOffset = 0

        bag_transformer = tf_bag.BagTfTransformer(bag)

        trans = tf.Transformer(True, rospy.Duration(10.0))




        for topic, msg, t in bag.read_messages():

            # observations
            # color
            if topic == '/camera/color/image_raw':
                rgb_arr = h5h.colorImageCallback(msg)
                colorList.append(rgb_arr)
                colorOffset += 1

                # cv2.imshow("Color Image", cv_image)
                # cv2.waitKey(1)

            # depth
            elif topic == 'camera/aligned_depth_to_color/image_raw':
                depth_arr = h5h.depthImageCallback(msg)
                depthList.append(depth_arr)
                depthOffset += 1

                # cv2.imshow("Depth Image", cv_image)
                # cv2.waitKey(1)


            # elif topic == '/camera/aligned_depth_to_color/image_raw/compressed':
                # try:
                #     data = np.frombuffer(msg.data, np.uint8)
                #     depth_image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                #     depth_image = cv2.resize(depth_image, (320, 180), interpolation=cv2.INTER_AREA)
                #     depth_arr = np.asarray(depth_image)

                #     depthList.append(depth_arr)
                #     depthOffset += 1

                #     # cv2.imshow("Depth Image", depth_image)
                #     # cv2.waitKey(1)
                # except CvBridgeError as e:
                #     rospy.logerr("CvBridge Error: {0}".format(e))


   
            # actions
            # gripper state
            elif topic == '/bariflex':
                # bariflex topic has strings with the data we want
                # regex to parse the "destination" of the gripper (the position wouldn't give us what we want)
                print(msg.data)
                regex = [float(x) for x in re.finditer(r"-{0,1}\d+\.\d+", msg.data)]
                bariList.append(regex[0])
                # saves
                uber_bari_arr.append(regex)
                bariOffset += 1

            
            # transforms
            elif topic == '/tf':
                try: 
                    # tf lookup
                    trans, quat = bag_transformer.lookupTransform('camera_link', 'map', t)

                    if(len(colorList) != 0 and len(depthList) != 0 and len(bariList) != 0):
                        
                        deltaTrans, deltaQuat = h5h.actionCallback(trans, quat)

                        uber_rgb_arr.append(colorList[-1*colorOffset])
                        colorOffset = 0
                        uber_depth_arr.append(depthList[-1*depthOffset])
                        depthOffset = 0
                        uber_action_arr.append(list.append[deltaTrans[0:3]. deltaQuat[0:4], bariList[-1*bariOffset]])
                        bariOffset = 0
                    else:
                        print("hasn't sampled enough")
                except Exception as e:
                    print(f"Exception {e}", file=sys.stderr)
                    continue
          
                    





        # creates the datasets
        print(np.array(uber_rgb_arr).shape)
        print(np.array(uber_depth_arr).shape)
        print(np.array(uber_action_arr).shape)
        print(np.array(uber_bari_arr).shape)
        
        color_dset = group.create_dataset(f"{name}: color images", data=np.array(uber_rgb_arr))
        depth_dset = group.create_dataset(f"{name}: depth images", data=np.array(uber_depth_arr))
        bari_dset = group.create_dataset(f"{name}: bariflex data", data=np.array(uber_bari_arr))
        action_dset = group.create_dataset(f"{name}: actions", data=np.array(uber_action_arr))