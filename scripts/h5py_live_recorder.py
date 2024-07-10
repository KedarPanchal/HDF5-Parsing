import json
import re
import sys
import os
import rosbag
import numpy as np
import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from tf2_msgs import TFMessage
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

import h5py_helper as h5h

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
bag_path = sys.argv[0]
hdf5_path = sys.argv[1]

script_dir = os.path.dirname(os.path.realpath(__file__))
script_folder_dir = os.path.dirname(script_dir)
data_dir = os.path.join(script_folder_dir, 'data')

color_list, depth_list, bari_list, action_list = [], [], [], []

color_offset, depth_offset, bari_offset = 0, 0, 0

uber_color_arr, uber_depth_arr, uber_bariflex_arr, uber_action_arr = np.empty(), np.empty(), np.empty(), np.empty()


def color_image_callback(data):
    global color_list, color_offset
    try:
        bridge = CvBridge()
        cv_image = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        cv_image = cv2.resize(cv_image, (320, 180), interpolation=cv2.INTER_AREA)
        rgb_arr = np.asarray(cv_image)

        color_list.append(rgb_arr)
        color_offset += 1

        # cv2.imshow("Color Image", cv_image)
        # cv2.waitKey(1)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

def depth_image_callback(data):
    global depth_list, depth_offset
    try:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(data, "16UC1")
        cv_image = cv2.resize(cv_image, (320, 180), interpolation=cv2.INTER_AREA)
        depth_arr = np.asarray(cv_image)

        depth_list.append(depth_arr)
        depth_offset += 1

        # cv2.imshow("Depth Image", cv_image)
        # cv2.waitKey(1)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

def bariflex_callback(data):
    global bari_list, bari_offset
    regex = [float(x.group()) for x in re.finditer(r"-{0,1}\d+\.\d+", data.data)]
    bari_list.append(regex)
    bari_offset += 1

def listener():
    rospy.init_node("hdf5_parser", anonymous=True)

    rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, color_image_callback)
    rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_image_callback)
    rospy.Subscriber("/bariflex", String, bariflex_callback)

    # tf lookup
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    rate = rospy.Rate(10)

    prev_pose = None
    while not rospy.is_shutdown():
        try:
            # tf lookup
            trans = tf_buffer.lookup_transform('camera_link', 'map', rospy.Time(0))
            translation = trans.transform.translation
            rotation = trans.transform.rotation

            # compute the 4x4 transform matrix representing the pose in the map
            mat = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
            transform_matrix = np.identity(4)
            transform_matrix[:3, :3] = mat[:3, :3]
            transform_matrix[0, 3] = translation.x
            transform_matrix[1, 3] = translation.y
            transform_matrix[2, 3] = translation.z
            # compute the relative pose between this pose and the previous pose, prev_pose
            if prev_pose is not None:
                inv_prev = inv(prev_pose)
                rel_pose = np.matmul(inv_prev, transform_matrix)
            else:
                rel_pose = transform_matrix
            prev_pose = transform_matrix
            translation = transform_matrix[:3, 3]
            quaternion = quaternion_from_matrix(transform_matrix)
            # record the relative pose together with the most recent depth and color image received by subscribers
            uber_color_arr.append[color_list[-1 * color_offset]]
            uber_depth_arr.append[depth_list[-1 * depth_offset]]
            uber_bariflex_arr.append[bari_list[-1 * bari_offset]]
            uber_action_arr.append[[translation, quaternion, bari_list[-1 * bari_offset][0]]]
            
            color_offset = 0
            depth_offset = 0
            bari_offset = 0
            
            
            
        except Exception as e:
            print("oops:", e)
        
        rate.sleep()

    rospy.spin()

if __name__ == "__main__":
    listener()
    
    with h5py.File(hdf5_path, "w") as hdf5_file:
        group = hdf5_file.create_group(bag_path)
        group.create_dataset(f"{bag_path}_color_images", data=np.array(uber_color_arr))
        group.create_dataset(f"{bag_path}_depth_images", data=np.array(uber_depth_arr))
        group.create_dataset(f"{bag_path}_bariflex_data", data=np.array(uber_bariflex_arr))
        group.create_dataset(f"{bag_path}_actions", data=np.array(uber_action_arr))