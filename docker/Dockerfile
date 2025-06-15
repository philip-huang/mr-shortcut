# Use ROS Noetic Robot base image
FROM osrf/ros:noetic-desktop-full-focal

# Install RViz and other dependencies
RUN apt-get update && apt-get install -y \
    ros-noetic-rviz \
    libopencv-dev \
    python3-opencv \
    python3-catkin-tools \
    python3-pip \
    ros-noetic-moveit \
    libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install software-properties-common and wget
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyYAML
RUN pip3 install pyyaml

# INSTALL Rviz and Moveit Visual, and control tools
RUN apt-get update && apt-get install -y \
    ros-noetic-rviz-visual-tools \
    ros-noetic-moveit-visual-tools \
    ros-noetic-ros-control \
    && rm -rf /var/lib/apt/lists/*

# Install pandas, numpy in pip
RUN pip3 install pandas numpy==1.24.4

# Setup environment
ENV HOME=/root
WORKDIR $HOME

# Create catkin workspace
RUN mkdir -p catkin_ws/src && \
    /bin/bash -c '. /opt/ros/noetic/setup.bash; cd catkin_ws; catkin init'

RUN cd catkin_ws/src \
    && git clone https://github.com/philip-huang/mr-shortcut.git \


RUN cd catkin_ws/src \
    && git clone https://github.com/intelligent-control-lab/Robot_Digital_Twin.git \
    && cd Robot_Digital_Twin \
    && git checkout dual-arm-mfi

# build the workspace
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; cd catkin_ws; catkin build -DCMAKE_BUILD_TYPE=Release'

# Source ROS setup file
RUN echo "source /opt/ros/noetic/setup.bash" >> $HOME/.bashrc && \
    echo "source $HOME/catkin_ws/devel/setup.bash" >> $HOME/.bashrc

# Set the entrypoint to the workspace
WORKDIR $HOME/catkin_ws
ENTRYPOINT ["/bin/bash"]
