#!/usr/bin/env python3

from controller_manager_msgs.srv import *
import geometry_msgs.msg
import numpy as np
import rospy

from active_grasp.bbox import AABBox, to_bbox_msg
from active_grasp.rviz import Visualizer
from active_grasp.srv import *
from robot_helpers.io import load_yaml
from robot_helpers.ros.conversions import to_pose_msg
from robot_helpers.ros.moveit import MoveItClient
from robot_helpers.ros.panda import PandaGripperClient
from robot_helpers.spatial import Transform


class HwNode:
    def __init__(self):
        self.load_parameters()
        self.init_robot_connection()
        self.init_visualizer()
        self.advertise_services()
        rospy.spin()

    def load_parameters(self):
        self.cfg = rospy.get_param("hw")
        self.T_base_roi = Transform.from_matrix(np.loadtxt(self.cfg["roi_calib_file"]))

    def init_robot_connection(self):
        self.gripper = PandaGripperClient()
        self.switch_controller = rospy.ServiceProxy(
            "controller_manager/switch_controller", SwitchController
        )
        self.moveit = MoveItClient("panda_arm")
        rospy.Timer(rospy.Duration(1), self.publish_table_co)

    def init_visualizer(self):
        self.vis = Visualizer()
        rospy.Timer(rospy.Duration(1), self.draw_bbox)

    def advertise_services(self):
        rospy.Service("seed", Seed, self.seed)
        rospy.Service("reset", Reset, self.reset)

    def seed(self, req):
        self.rng = np.random.default_rng(req.seed)
        rospy.loginfo(f"Seeded the rng with {req.seed}.")
        return SeedResponse()

    def reset(self, req):
        q0, bbox = self.load_config()

        # Move to the initial configuration
        self.switch_to_joint_trajectory_controller()
        q0 += self.rng.uniform(-0.069, 0.069, 7)
        self.moveit.goto(q0, velocity_scaling=0.4)
        self.gripper.move(0.08)

        return ResetResponse(to_bbox_msg(bbox))

    def load_config(self):
        scene_config = load_yaml(self.cfg["scene_file"])
        q0 = scene_config["q0"]
        bbox_min = self.T_base_roi.apply(scene_config["target"]["min"])
        bbox_max = self.T_base_roi.apply(scene_config["target"]["max"])
        bbox = AABBox(bbox_min, bbox_max)
        return q0, bbox

    def switch_to_joint_trajectory_controller(self):
        req = SwitchControllerRequest()
        req.start_controllers = ["position_joint_trajectory_controller"]
        req.stop_controllers = ["cartesian_velocity_controller"]
        req.strictness = 1
        self.switch_controller(req)

    def draw_bbox(self, event):
        _, bbox = self.load_config()
        self.vis.bbox("panda_link0", bbox)

    def publish_table_co(self, event):
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = "panda_link0"
        msg.pose = to_pose_msg(self.T_base_roi * Transform.t_[0.15, 0.15, 0.005])
        self.moveit.scene.add_box("table", msg, size=(0.8, 0.8, 0.01))


def main():
    rospy.init_node("hw")
    HwNode()


if __name__ == "__main__":
    main()
