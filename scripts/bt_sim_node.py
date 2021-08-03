#!/usr/bin/env python3

from actionlib import SimpleActionServer
import cv_bridge
from franka_gripper.msg import *
from geometry_msgs.msg import PoseStamped
import numpy as np
import rospy
from sensor_msgs.msg import JointState, Image, CameraInfo
from threading import Thread
import tf2_ros

from active_grasp.bbox import to_bbox_msg
from active_grasp.srv import *
from active_grasp.simulation import Simulation
from robot_helpers.ros.conversions import *


class BtSimNode:
    def __init__(self):
        self.gui = rospy.get_param("~gui", True)
        self.sim = Simulation(gui=self.gui)
        self.init_plugins()
        self.advertise_services()

    def init_plugins(self):
        self.plugins = [
            PhysicsPlugin(self.sim),
            JointStatePlugin(self.sim.arm, self.sim.gripper),
            ArmControllerPlugin(self.sim.arm, self.sim.controller),
            MoveActionPlugin(self.sim.gripper),
            GraspActionPlugin(self.sim.gripper),
            CameraPlugin(self.sim.camera),
        ]

    def advertise_services(self):
        rospy.Service("seed", Seed, self.seed)
        rospy.Service("reset", Reset, self.reset)

    def seed(self, req):
        self.sim.seed(req.seed)
        return SeedResponse()

    def reset(self, req):
        for plugin in self.plugins:
            plugin.is_running = False
        rospy.sleep(1.0)  # TODO replace with a read-write lock
        bbox = self.sim.reset()
        res = ResetResponse(to_bbox_msg(bbox))
        for plugin in self.plugins:
            plugin.is_running = True
        return res

    def run(self):
        self.start_plugins()
        rospy.spin()

    def start_plugins(self):
        for plugin in self.plugins:
            plugin.thread.start()
            plugin.is_running = True


class Plugin:
    """A plugin that spins at a constant rate in its own thread."""

    def __init__(self, rate):
        self.rate = rate
        self.thread = Thread(target=self.loop, daemon=True)
        self.is_running = False

    def loop(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if self.is_running:
                self.update()
            rate.sleep()

    def update(self):
        raise NotImplementedError


class PhysicsPlugin(Plugin):
    def __init__(self, sim):
        super().__init__(sim.rate)
        self.sim = sim

    def update(self):
        self.sim.step()


class JointStatePlugin(Plugin):
    def __init__(self, arm, gripper, rate=30):
        super().__init__(rate)
        self.arm = arm
        self.gripper = gripper
        self.pub = rospy.Publisher("joint_states", JointState, queue_size=10)

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
        self.pub.publish(msg)


class ArmControllerPlugin(Plugin):
    def __init__(self, arm, controller, rate=30):
        super().__init__(rate)
        self.arm = arm
        self.controller = controller
        rospy.Subscriber("command", PoseStamped, self.target_cb)

    def target_cb(self, msg):
        assert msg.header.frame_id == self.arm.base_frame
        self.controller.x_d = from_pose_msg(msg.pose)

    def update(self):
        q, _ = self.arm.get_state()
        cmd = self.controller.update(q)
        self.arm.set_desired_joint_velocities(cmd)


class MoveActionPlugin(Plugin):
    def __init__(self, gripper, rate=10):
        super().__init__(rate)
        self.gripper = gripper
        self.dt = 1.0 / self.rate
        self.init_action_server()

    def init_action_server(self):
        name = "/franka_gripper/move"
        self.action_server = SimpleActionServer(name, MoveAction, auto_start=False)
        self.action_server.register_goal_callback(self.action_goal_cb)
        self.action_server.start()

    def action_goal_cb(self):
        self.elapsed_time = 0.0
        goal = self.action_server.accept_new_goal()
        self.gripper.set_desired_width(goal.width)

    def update(self):
        if self.action_server.is_active():
            self.elapsed_time += self.dt
            if self.elapsed_time > 1.0:
                self.action_server.set_succeeded()


class GraspActionPlugin(Plugin):
    def __init__(self, gripper, rate=10):
        super().__init__(rate)
        self.gripper = gripper
        self.dt = 1.0 / self.rate
        self.init_action_server()

    def init_action_server(self):
        name = "/franka_gripper/grasp"
        self.action_server = SimpleActionServer(name, GraspAction, auto_start=False)
        self.action_server.register_goal_callback(self.action_goal_cb)
        self.action_server.start()

    def action_goal_cb(self):
        self.elapsed_time = 0.0
        goal = self.action_server.accept_new_goal()
        self.gripper.set_desired_width(goal.width)

    def update(self):
        if self.action_server.is_active():
            self.elapsed_time += self.dt
            if self.elapsed_time > 1.0:
                self.action_server.set_succeeded()


class CameraPlugin(Plugin):
    def __init__(self, camera, name="camera", rate=5):
        super().__init__(rate)
        self.camera = camera
        self.name = name
        self.cv_bridge = cv_bridge.CvBridge()
        self.init_publishers()

    def init_publishers(self):
        topic = self.name + "/depth/camera_info"
        self.info_pub = rospy.Publisher(topic, CameraInfo, queue_size=10)
        topic = self.name + "/depth/image_raw"
        self.depth_pub = rospy.Publisher(topic, Image, queue_size=10)

    def update(self):
        stamp = rospy.Time.now()

        msg = to_camera_info_msg(self.camera.intrinsic)
        msg.header.frame_id = self.name + "_optical_frame"
        msg.header.stamp = stamp
        self.info_pub.publish(msg)

        _, depth, _ = self.camera.get_image()
        msg = self.cv_bridge.cv2_to_imgmsg(depth)
        msg.header.stamp = stamp
        self.depth_pub.publish(msg)


def main():
    rospy.init_node("bt_sim")
    server = BtSimNode()
    server.run()


if __name__ == "__main__":
    main()
