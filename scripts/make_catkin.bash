#!/bin/bash

source /opt/ros/noetic/setup.bash
mkdir -p /usr/share/hdf5_parse/src/
cd /usr/share/hdf5_parse/src/
[ -d tf_bag ] && rm -rf tf_bag
git clone "https://github.com/IFL-CAMP/tf_bag.git"
cd ..
rosdep install -ryi --from-paths . --ignore-src
catkin_make
