#!/usr/bin/env python3

import actionlib
import argparse
import cv_bridge
import franka_gripper.msg
from geometry_msgs.msg import PoseStamped
import numpy as np
import rospy
from sensor_msgs.msg import JointState, Image, CameraInfo
import std_srvs.srv as std_srvs
import tf2_ros

import active_grasp.srv
from robot_utils.ros.conversions import *
from simulation import Simulation
from utils import *


class BtSimNode:
    def __init__(self, gui):
        self.sim = Simulation(gui=gui)
        self.robot_state_interface = RobotStateInterface(self.sim.arm, self.sim.gripper)
        self.arm_interface = ArmInterface(self.sim.arm, self.sim.controller)
        self.gripper_interface = GripperInterface(self.sim.gripper)
        self.camera_interface = CameraInterface(self.sim.camera)
        self.step_cnt = 0
        self.reset_requested = False

        self.advertise_services()
        self.broadcast_transforms()

    def advertise_services(self):
        rospy.Service("reset", active_grasp.srv.Reset, self.reset)

    def broadcast_transforms(self):
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        msgs = [
            to_transform_stamped_msg(self.sim.T_W_B, "world", "panda_link0"),
            to_transform_stamped_msg(
                Transform.translation(self.sim.origin), "world", "task"
            ),
        ]
        self.static_broadcaster.sendTransform(msgs)

    def reset(self, req):
        self.reset_requested = True
        rospy.sleep(1.0)  # wait for the latest sim step to finish
        bbox = self.sim.reset()
        res = active_grasp.srv.ResetResponse(to_bbox_msg(bbox))
        self.step_cnt = 0
        self.reset_requested = False
        return res

    def run(self):
        rate = rospy.Rate(self.sim.rate)
        while not rospy.is_shutdown():
            if not self.reset_requested:
                self.handle_updates()
                self.sim.step()
                self.step_cnt = (self.step_cnt + 1) % self.sim.rate
            rate.sleep()

    def handle_updates(self):
        self.robot_state_interface.update()
        self.arm_interface.update()
        self.gripper_interface.update(self.sim.dt)
        if self.step_cnt % int(self.sim.rate / 5) == 0:
            self.camera_interface.update()


class RobotStateInterface:
    def __init__(self, arm, gripper):
        self.arm = arm
        self.gripper = gripper
        self.joint_pub = rospy.Publisher("joint_states", JointState, queue_size=10)

    def update(self):
        q, _ = self.arm.get_state()
        width = self.gripper.read()
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ["panda_joint{}".format(i) for i in range(1, 8)] + [
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]
        msg.position = np.r_[q, 0.5 * width, 0.5 * width]
        self.joint_pub.publish(msg)


class ArmInterface:
    def __init__(self, arm, controller):
        self.arm = arm
        self.controller = controller
        rospy.Subscriber("command", PoseStamped, self.target_cb)

    def update(self):
        q, _ = self.arm.get_state()
        cmd = self.controller.update(q)
        self.arm.set_desired_joint_velocities(cmd)

    def target_cb(self, msg):
        assert msg.header.frame_id == self.arm.base_frame
        self.controller.x_d = from_pose_msg(msg.pose)


class GripperInterface:
    def __init__(self, gripper):
        self.gripper = gripper
        self.move_server = actionlib.SimpleActionServer(
            "/franka_gripper/move",
            franka_gripper.msg.MoveAction,
            auto_start=False,
        )
        self.move_server.register_goal_callback(self.move_action_goal_cb)
        self.move_server.start()

        self.grasp_server = actionlib.SimpleActionServer(
            "/franka_gripper/grasp",
            franka_gripper.msg.GraspAction,
            auto_start=False,
        )
        self.grasp_server.register_goal_callback(self.grasp_action_goal_cb)
        self.grasp_server.start()

    def move_action_goal_cb(self):
        self.elapsed_time_since_move_action_goal = 0.0
        goal = self.move_server.accept_new_goal()
        self.gripper.set_desired_width(goal.width)

    def grasp_action_goal_cb(self):
        self.elapsed_time_since_grasp_action_goal = 0.0
        goal = self.grasp_server.accept_new_goal()
        self.gripper.set_desired_width(goal.width)

    def update(self, dt):
        if self.move_server.is_active():
            self.elapsed_time_since_move_action_goal += dt
            if self.elapsed_time_since_move_action_goal > 1.0:
                self.move_server.set_succeeded()
        if self.grasp_server.is_active():
            self.elapsed_time_since_grasp_action_goal += dt
            if self.elapsed_time_since_grasp_action_goal > 1.0:
                self.grasp_server.set_succeeded()


class CameraInterface:
    def __init__(self, camera):
        self.camera = camera
        self.cv_bridge = cv_bridge.CvBridge()
        self.cam_info_msg = to_camera_info_msg(self.camera.intrinsic)
        self.cam_info_msg.header.frame_id = "cam_optical_frame"
        self.cam_info_pub = rospy.Publisher(
            "/cam/depth/camera_info",
            CameraInfo,
            queue_size=10,
        )
        self.depth_pub = rospy.Publisher("/cam/depth/image_raw", Image, queue_size=10)

    def update(self):
        stamp = rospy.Time.now()
        self.cam_info_msg.header.stamp = stamp
        self.cam_info_pub.publish(self.cam_info_msg)
        img = self.camera.get_image()
        depth_msg = self.cv_bridge.cv2_to_imgmsg(img.depth)
        depth_msg.header.stamp = stamp
        self.depth_pub.publish(depth_msg)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    return parser


def main():
    rospy.init_node("bt_sim")
    parser = create_parser()
    args, _ = parser.parse_known_args()
    server = BtSimNode(args.gui)
    server.run()


if __name__ == "__main__":
    main()
