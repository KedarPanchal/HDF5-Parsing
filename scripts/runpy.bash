#! /bin/bash

if [[ -d scripts ]]; then
    source ../devel/setup.bash
else
    source ./devel/setup.bash
fi

python3 "$@"
