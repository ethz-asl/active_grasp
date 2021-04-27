#!/usr/bin/env python3

import argparse

import actionlib
import cv_bridge
import numpy as np
import rospy

import franka_gripper.msg
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState, Image
import std_srvs.srv

from robot_utils.controllers import CartesianPoseController
from robot_utils.ros.conversions import *
from robot_utils.ros.tf import TransformTree

from simulation import BtPandaEnv

CONTROLLER_UPDATE_RATE = 60
JOINT_PUBLISH_RATE = 60
CAM_PUBLISH_RATE = 5


class BtSimNode:
    def __init__(self, gui):
        self.sim = BtPandaEnv(gui=gui, sleep=False)
        self.controller = CartesianPoseController(self.sim.arm)
        self.cv_bridge = cv_bridge.CvBridge()
        self.tf_tree = TransformTree()
        self.reset_server = rospy.Service("reset", std_srvs.srv.Trigger, self.reset)
        self.move_server = actionlib.SimpleActionServer(
            "move",
            franka_gripper.msg.MoveAction,
            self.move,
            False,
        )
        self.joint_pub = rospy.Publisher("joint_states", JointState, queue_size=10)
        self.color_pub = rospy.Publisher("/cam/color/image_raw", Image, queue_size=10)
        self.depth_pub = rospy.Publisher("/cam/depth/image_raw", Image, queue_size=10)
        rospy.Subscriber("target", Pose, self.target_pose_cb)
        self.step_cnt = 0
        self.reset_requested = False
        self.move_server.start()
        T_base_task = Transform(Rotation.identity(), np.r_[0.2, -0.1, 0.1])
        self.tf_tree.broadcast_static(T_base_task, "panda_link0", "task")

    def reset(self, req):
        self.reset_requested = True
        rospy.sleep(1.0)  # wait for the latest sim step to finish
        self.sim.reset()
        self.controller.set_target(self.sim.arm.pose())
        self.step_cnt = 0
        self.reset_requested = False
        return std_srvs.srv.TriggerResponse(success=True)

    def run(self):
        rate = rospy.Rate(self.sim.rate)
        while not rospy.is_shutdown():
            if not self.reset_requested:
                self.handle_updates()
                self.sim.step()
                self.step_cnt = (self.step_cnt + 1) % self.sim.rate
            rate.sleep()

    def handle_updates(self):
        if self.step_cnt % int(self.sim.rate / CONTROLLER_UPDATE_RATE) == 0:
            self.controller.update()
        if self.step_cnt % int(self.sim.rate / JOINT_PUBLISH_RATE) == 0:
            self.publish_joint_state()
        if self.step_cnt % int(self.sim.rate / CAM_PUBLISH_RATE) == 0:
            self.publish_cam_imgs()

    def publish_joint_state(self):
        q, dq = self.sim.arm.get_state()
        width = self.sim.gripper.read()
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ["panda_joint{}".format(i) for i in range(1, 8)] + [
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]
        msg.position = np.r_[q, 0.5 * width, 0.5 * width]
        msg.velocity = dq
        self.joint_pub.publish(msg)

    def publish_cam_imgs(self):
        color, depth = self.sim.camera.update()
        color_msg = self.cv_bridge.cv2_to_imgmsg(color, encoding="rgb8")
        self.color_pub.publish(color_msg)
        depth_msg = self.cv_bridge.cv2_to_imgmsg(depth)
        self.depth_pub.publish(depth_msg)

    def move(self, goal):
        self.sim.gripper.move(goal.width)
        self.move_server.set_succeeded()

    def target_pose_cb(self, msg):
        self.controller.set_target(from_pose_msg(msg))


def main(args):
    rospy.init_node("bt_sim")
    server = BtSimNode(args.gui)
    server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args, _ = parser.parse_known_args()
    main(args)
