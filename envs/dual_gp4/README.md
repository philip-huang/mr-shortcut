# Robot_Digital_Twin
This repository includes the digital twin models for
* Fanuc LR-mate 200id-7L.
* Yaskawa GP4.
* Yaskawa GP50.
* Lego bricks, including Lego baseplates, 1x1, 1x2, 1x4, 1x6, 1x8, 2x2, 2x4, 2x6, 2x8.

## Robot Rviz visualization
```
roslaunch robot_digital_twin YOUR_ROBOT_rviz.launch
```

## Gazebo Usage
1. Create your digital twin launch file.
2. Launch the Gazebo world.
```
roslaunch robot_digital_twin your_launch_file.launch
```

### Examples:
1. Single Fanuc arm environment with Lego bricks.
```
roslaunch robot_digital_twin single_fanuc_lego.launch
```
2. Dual GP4 arms environment.
```
roslaunch robot_digital_twin dual_gp4.launch
```

## Lego Visualization
1. Specify the Lego structure in `./scripts/task_graph.json`.
2. Launch the environment
```
roslaunch robot_digital_twin vis_lego.launch num_bx:="number of bricks you want" color_bx:="color you want"
```
Example
```
roslaunch robot_digital_twin vis_lego.launch num_b2:=10 color_b2:=Red
```
3. Run script
```
python3 ./scripts/vis_lego.py
```

## Citation
If you find this repository helpful, please kindly cite our work.
```
@article{liu2023simulationaided,
  title={Simulation-aided Learning from Demonstration for Robotic LEGO Construction}, 
  author={Ruixuan Liu and Alan Chen and Xusheng Luo and Changliu Liu},
  journal={arXiv preprint arXiv:2309.11010},
  year={2023}
}
```

## License
This project is licensed under the MIT License.
