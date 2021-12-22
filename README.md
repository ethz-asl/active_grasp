# active_grasp

## Setup

### Hardware

While the policies are hardware-agnostic, the experiments are designed to work with a Franka Emika Panda and an Intel Realsense D435 attached to the wrist of the robot.

### Software

The code was developed and tested on Ubuntu 20.04 with ROS Noetic. It depends on the following external packages:

- [MoveIt](https://github.com/ros-planning/panda_moveit_config.git)
- [TRAC-IK](http://wiki.ros.org/trac_ik)
- [VGN](https://github.com/ethz-asl/vgn)
- `franka_ros` and `realsense2_camera` (only required for hardware experiments)

Additional Python dependencies can be installed with

```
pip install -r requirements.txt
```

Finally, run `catkin build active_grasp` to build the package.

## Experiments

Start a roscore.

```
roscore
```

To run simulated grasping experiments.

```
# Start the simulated environment
mon launch active_grasp env.launch sim:=true

# Run the grasping experiment
python3 scripts/run.py nbv
```

To run real-robot grasping experiments.

```
# Start the hardware drivers
mon launch active_grasp hw.launch

# Launch the hw environment
mon launch active_grasp env.launch sim:=false

# Run the grasping experiment
python3 scripts/run.py nbv --wait-for-input
```
