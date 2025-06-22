#!/bin/bash

# Get the directory of this script and the parent directory (repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# allow any application to connect to the X server
xhost +local:docker

# run docker with x11 forwarding and mount the current repo
sudo docker run -it \
    --gpus all \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$REPO_ROOT:/root/catkin_ws/src/mr-shortcut" \
    --name=mr_shortcut1 \
    mr-shortcut-image
