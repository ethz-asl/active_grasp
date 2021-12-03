#!/usr/bin/env python3

from controller_manager_msgs.srv import *
import numpy as np
import rospy

from active_grasp.bbox import AABBox, to_bbox_msg
from active_grasp.srv import *
from robot_helpers.io import load_yaml
from robot_helpers.ros.moveit import MoveItClient
from robot_helpers.ros.panda import PandaGripperClient
from robot_helpers.spatial import Transform


class HwNode:
    def __init__(self):
        self.load_parameters()
        self.init_robot_connection()
        self.advertise_services()
        rospy.spin()

    def load_parameters(self):
        cfg = rospy.get_param("hw")
        self.T_base_roi = Transform.from_matrix(np.loadtxt(cfg["roi_calib_file"]))
        self.scene_config = load_yaml(cfg["scene_file"])

    def init_robot_connection(self):
        self.gripper = PandaGripperClient()
        self.switch_to_joint_trajectory_controller = rospy.ServiceProxy(
            "controller_manager/switch_controller", SwitchController
        )
        self.moveit = MoveItClient("panda_arm")

    def advertise_services(self):
        rospy.Service("seed", Seed, self.seed)
        rospy.Service("reset", Reset, self.reset)

    def seed(self, req):
        # Nothing to do
        return SeedResponse()

    def reset(self, req):
        self.gripper.move(0.04)

        # Move to the initial configuration
        self.switch_to_joint_trajectory_controller()
        self.moveit.goto(self.scene_config["q0"])

        # Construct bounding box
        bbox_min = self.T_base_roi.apply(self.scene_config["target"]["min"])
        bbox_max = self.T_base_roi.apply(self.scene_config["target"]["max"])
        bbox = AABBox(bbox_min, bbox_max)

        return ResetResponse(to_bbox_msg(bbox))

    def switch_to_joint_trajectory_controller(self):
        req = SwitchControllerRequest()
        req.start_controllers = ["position_joint_trajectory_controller"]
        req.stop_controllers = ["cartesian_velocity_controller"]
        req.strictness = 1
        self.switch_controller(req)


def main():
    rospy.init_node("hw")
    HwNode()


if __name__ == "__main__":
    main()
