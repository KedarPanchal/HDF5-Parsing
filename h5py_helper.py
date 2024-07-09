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

prev_pose = None
deltaTrans = [0, 0, 0]
deltaQuat = [0, 0, 0, 1]


def colorImageCallback(data):
    try:
        bridge = CvBridge()
        cv_image = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        cv_image = cv2.resize(cv_image, (320, 180), interpolation=cv2.INTER_AREA)
        rgb_arr = np.asarray(cv_image)

        return rgb_arr

        # cv2.imshow("Color Image", cv_image)
        # cv2.waitKey(1)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

def depthImageCallback(data):
     # depth
    try:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(data, "16UC1")
        cv_image = cv2.resize(cv_image, (320, 180), interpolation=cv2.INTER_AREA)
        depth_arr = np.asarray(cv_image)

        return depth_arr

        # cv2.imshow("Depth Image", cv_image)
        # cv2.waitKey(1)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

def actionCallback(trans, quat):
    mat = quaternion_matrix(quat)
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = mat[:3, :3]
    transform_matrix[0, 3] = trans[0]
    transform_matrix[1, 3] = trans[1]
    transform_matrix[2, 3] = trans[2]


    if(prev_pose is not None):
        inv_prev = inv(prev_pose)
        rel_pose = np.matmul(inv_prev, transform_matrix)
    else:
        rel_pose = transform_matrix
    prev_pose = transform_matrix

    rel_pose[3, 0] = deltaTrans[0]
    rel_pose[3, 1] = deltaTrans[1]
    rel_pose[3, 2] = deltaTrans[2]

    deltaQuat = quaternion_from_matrix(rel_pose)

    return deltaTrans, deltaQuat