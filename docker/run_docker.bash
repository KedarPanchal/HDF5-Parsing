#!/bin/bash

docker build -t "hdf5_parse" docker/
if [[ -n $1 ]]; then
    docker run -v "$(pwd)/:/usr/share/hdf5_parse" --name "$1" -it "hdf5_parse"
else
    docker run -v "$(pwd)/:/usr/share/hdf5_parse" -it "hdf5_parse"
fi