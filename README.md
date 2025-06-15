# Multi-Robot Shortcut Benchmark

This is the code repository for our paper "Benchmarking Shortcutting Techniques for Multi-Robot Arm Motion Planning".

## Installation
Build the docker image and run it inside docker
```
cd docker && bash build.sh
```

If you are not using the docker file, the following setup has been tested on Ubuntu 20.04 with ROS Noetic. You may need to install some system dependencies
- [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)
- [moveit](https://moveit.ai/install/)
- [catkin tools](https://catkin-tools.readthedocs.io/en/latest/)
- [rviz tools](http://wiki.ros.org/rviz_visual_tools)
- [moveit visual tools](http://wiki.ros.org/moveit_visual_tools)


Follow the ros tutorial to create a workspace [tutorial](https://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment)
For building code, I use catkin tools, which are documented [here](https://catkin-tools.readthedocs.io/en/latest/)

Once you have download ros, and other system deps, under your catkin_ws/src, download these to your workspace
- [moveit configs](https://github.com/philip-huang/moveit_configs) 
- [gp4 digital twin](https://github.com/intelligent-control-lab/Robot_Digital_Twin.git). checkout to the ``dual_arm_gp4`` branch!
- this repo

then do ```catkin build mr``` to compile


## Benchmark planner on different environments

I have included several launch files for running planner in different environments, which are panda_two, panda_three, panda_four, panda_two_rod, and panda_four_bin. To benchmark the performance of the planner / shortcutting algorithm, you can run
```
roslaunch mr-shortcut panda_two.launch benchmark:=true
```


## Code Structure

- `include`: API of the library
    - `instance.h`: Class for the planning scene
    - `logger.h`: Utilities for logging
    - `planner.h`: Implements a multi-robot planning interface
    - `SingleAgentPlanner.h`: Implements the single agent planning algorithm
    - `tpg.h`: Implements the Temporal Plan Graph execution policy and post-processing algorithm

- `src`: Code for the library and executable
    - `demo_node.cpp`: Executable for testing single-step planning

- `launch`: 
    - `dual_gp4.launch`, `panda_two.launch`, `panda_two_rod.launch`, `panda_three.launch`, `panda_four.launch`, `panda_four_bins.launch`: Launch files for testing the single agent planning

- `scripts`:
    - `benchmark.py`: Python scripts for benchmarking motion planning/TPG processing in parallel
    - `plot.py`: Visualize the results 
