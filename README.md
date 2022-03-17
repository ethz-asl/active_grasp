# active_grasp

Accompanying code for our IROS 2022 submission: Closed-Loop Next-Best-View Planning for Target-Driven Grasping.

## Setup

While the policies are hardware-agnostic, the experiments are designed to work with a Franka Emika Panda and an Intel Realsense D435 attached to the wrist of the robot.

The code was developed and tested on Ubuntu 20.04 with ROS Noetic. It depends on the following external packages:

- [MoveIt](https://github.com/ros-planning/panda_moveit_config)
- [robot_helpers](https://github.com/mbreyer/robot_helpers)
- [TRAC-IK](http://wiki.ros.org/trac_ik)
- [VGN](https://github.com/ethz-asl/vgn/tree/devel)
- franka_ros and realsense2_camera (only required for hardware experiments)

Additional Python dependencies can be installed with

```
pip install -r requirements.txt
```

Download the [assets folder](https://drive.google.com/file/d/19NqFOrHaICXdT9NwmHSlHqWVlDDMGyeb/view) and place it inside the cloned repository.

Finally, run `catkin build active_grasp` to build the package.

## Experiments

Start a roscore.

```
roscore
```

To run simulation experiments.

```
# Start the simulation environment
roslaunch active_grasp env.launch sim:=true

# Run the experiment
python3 scripts/run.py nbv
```

To run real world experiments.

```
# Start the hardware drivers
roslaunch active_grasp hw.launch

# Launch the hw environment
roslaunch active_grasp env.launch sim:=false

# Run the experiment
python3 scripts/run.py nbv --wait-for-input
```
