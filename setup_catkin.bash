#!/bin/bash

cd /usr/share/hdf5_parse
mkdir src
cd src
git clone https://github.com/IFL-CAMP/tf_bag.git
cd ..
rosdep install -ryi --from-paths . --ignore-src
catkin_make