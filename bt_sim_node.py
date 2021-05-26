#!/usr/bin/env python3

import actionlib
import argparse
import cv_bridge
import numpy as np
import rospy

import franka_gripper.msg
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState, Image, CameraInfo
import std_srvs.srv

from robot_tools.controllers import CartesianPoseController
from robot_tools.ros.conversions import *
from robot_tools.ros.tf import TransformTree
from simulation import Simulation


class BtSimNode:
    def __init__(self, gui):
        self.sim = Simulation(gui=gui, sleep=False)
        self.controller = CartesianPoseController(self.sim.arm)

        self.controller_update_rate = 60
        self.joint_state_publish_rate = 60
        self.camera_publish_rate = 5

        self.cv_bridge = cv_bridge.CvBridge()
        self.tf_tree = TransformTree()

        self.setup_robot_topics()
        self.setup_camera_topics()
        self.setup_gripper_actions()
        self.broadcast_transforms()

        rospy.Service("reset", std_srvs.srv.Trigger, self.reset)
        self.step_cnt = 0
        self.reset_requested = False

    def setup_robot_topics(self):
        self.joint_pub = rospy.Publisher("joint_states", JointState, queue_size=10)
        rospy.Subscriber("cmd", Pose, self.target_pose_cb)

    def target_pose_cb(self, msg):
        self.controller.set_target(from_pose_msg(msg))

    def setup_camera_topics(self):
        self.cam_info_msg = to_camera_info_msg(self.sim.camera.intrinsic)
        self.cam_info_msg.header.frame_id = "cam_optical_frame"
        self.cam_info_pub = rospy.Publisher(
            "/cam/depth/camera_info",
            CameraInfo,
            queue_size=10,
        )
        self.depth_pub = rospy.Publisher("/cam/depth/image_raw", Image, queue_size=10)

    def setup_gripper_actions(self):
        self.move_server = actionlib.SimpleActionServer(
            "/franka_gripper/move",
            franka_gripper.msg.MoveAction,
            self.move,
            False,
        )
        self.move_server.start()

    def move(self, goal):
        self.sim.gripper.move(goal.width)
        self.move_server.set_succeeded()

    def broadcast_transforms(self):
        msgs = []
        msg = geometry_msgs.msg.TransformStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.child_frame_id = "panda_link0"
        msg.transform = to_transform_msg(self.sim.T_W_B)
        msgs.append(msg)

        msg = geometry_msgs.msg.TransformStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.child_frame_id = "task"
        msg.transform = to_transform_msg(Transform.translation(self.sim.origin))

        msgs.append(msg)

        self.tf_tree.static_broadcaster.sendTransform(msgs)

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
        if self.step_cnt % int(self.sim.rate / self.controller_update_rate) == 0:
            self.controller.update()
        if self.step_cnt % int(self.sim.rate / self.joint_state_publish_rate) == 0:
            self.publish_joint_state()
        if self.step_cnt % int(self.sim.rate / self.camera_publish_rate) == 0:
            self.publish_cam_info()
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

    def publish_cam_info(self):
        self.cam_info_msg.header.stamp = rospy.Time.now()
        self.cam_info_pub.publish(self.cam_info_msg)

    def publish_cam_imgs(self):
        _, depth = self.sim.camera.get_image()
        depth_msg = self.cv_bridge.cv2_to_imgmsg(depth)
        depth_msg.header.stamp = rospy.Time.now()
        self.depth_pub.publish(depth_msg)


def main(args):
    rospy.init_node("bt_sim")
    server = BtSimNode(args.gui)
    server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args, _ = parser.parse_known_args()
    main(args)
