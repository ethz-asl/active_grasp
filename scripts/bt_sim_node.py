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

from active_grasp.srv import Reset, ResetResponse
from active_grasp.simulation import Simulation
from active_grasp.utils import *
from robot_utils.ros.conversions import *


class BtSimNode:
    def __init__(self):
        self.gui = rospy.get_param("~gui", True)
        seed = rospy.get_param("~seed", None)

        rng = np.random.default_rng(seed) if seed else np.random
        self.sim = Simulation(gui=self.gui, rng=rng)

        self._init_plugins()
        self._advertise_services()
        self._broadcast_transforms()

    def _init_plugins(self):
        self.plugins = [
            PhysicsPlugin(self.sim),
            JointStatePlugin(self.sim.arm, self.sim.gripper),
            ArmControllerPlugin(self.sim.arm, self.sim.controller),
            GripperControllerPlugin(self.sim.gripper),
            CameraPlugin(self.sim.camera),
        ]

    def _advertise_services(self):
        rospy.Service("reset", Reset, self.reset)

    def _broadcast_transforms(self):
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        msgs = [
            to_transform_stamped_msg(self.sim.T_W_B, "world", "panda_link0"),
            to_transform_stamped_msg(
                Transform.translation(self.sim.origin), "world", "task"
            ),
        ]
        self.static_broadcaster.sendTransform(msgs)

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
        self._start_plugins()
        rospy.spin()

    def _start_plugins(self):
        for plugin in self.plugins:
            plugin.thread.start()
            plugin.is_running = True


class Plugin:
    """A plugin that spins at a constant rate in its own thread."""

    def __init__(self, rate):
        self.rate = rate
        self.thread = Thread(target=self._loop, daemon=True)
        self.is_running = False

    def _loop(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if self.is_running:
                self._update()
            rate.sleep()

    def _update(self):
        raise NotImplementedError


class PhysicsPlugin(Plugin):
    def __init__(self, sim):
        super().__init__(sim.rate)
        self.sim = sim

    def _update(self):
        self.sim.step()


class JointStatePlugin(Plugin):
    def __init__(self, arm, gripper, rate=30):
        super().__init__(rate)
        self.arm = arm
        self.gripper = gripper
        self.pub = rospy.Publisher("joint_states", JointState, queue_size=10)

    def _update(self):
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
        rospy.Subscriber("command", PoseStamped, self._target_cb)

    def _target_cb(self, msg):
        assert msg.header.frame_id == self.arm.base_frame
        self.controller.x_d = from_pose_msg(msg.pose)

    def _update(self):
        q, _ = self.arm.get_state()
        cmd = self.controller.update(q)
        self.arm.set_desired_joint_velocities(cmd)


class GripperControllerPlugin(Plugin):
    def __init__(self, gripper, rate=10):
        super().__init__(rate)
        self.gripper = gripper
        self.dt = 1.0 / self.rate
        self._init_move_action_server()
        self._init_grasp_action_server()

    def _init_move_action_server(self):
        name = "/franka_gripper/move"
        self.move_server = SimpleActionServer(name, MoveAction, auto_start=False)
        self.move_server.register_goal_callback(self._move_action_goal_cb)
        self.move_server.start()

    def _init_grasp_action_server(self):
        name = "/franka_gripper/grasp"
        self.grasp_server = SimpleActionServer(name, GraspAction, auto_start=False)
        self.grasp_server.register_goal_callback(self._grasp_action_goal_cb)
        self.grasp_server.start()

    def _move_action_goal_cb(self):
        self.elapsed_time_since_move_action_goal = 0.0
        goal = self.move_server.accept_new_goal()
        self.gripper.set_desired_width(goal.width)

    def _grasp_action_goal_cb(self):
        self.elapsed_time_since_grasp_action_goal = 0.0
        goal = self.grasp_server.accept_new_goal()
        self.gripper.set_desired_width(goal.width)

    def _update(self):
        if self.move_server.is_active():
            self.elapsed_time_since_move_action_goal += self.dt
            if self.elapsed_time_since_move_action_goal > 1.0:
                self.move_server.set_succeeded()
        if self.grasp_server.is_active():
            self.elapsed_time_since_grasp_action_goal += self.dt
            if self.elapsed_time_since_grasp_action_goal > 1.0:
                self.grasp_server.set_succeeded()


class CameraPlugin(Plugin):
    def __init__(self, camera, name="camera", rate=10):
        super().__init__(rate)
        self.camera = camera
        self.name = name
        self.cv_bridge = cv_bridge.CvBridge()
        self._init_publishers()

    def _init_publishers(self):
        topic = self.name + "/depth/camera_info"
        self.info_pub = rospy.Publisher(topic, CameraInfo, queue_size=10)
        topic = self.name + "/depth/image_raw"
        self.depth_pub = rospy.Publisher(topic, Image, queue_size=10)

    def _update(self):
        stamp = rospy.Time.now()

        msg = to_camera_info_msg(self.camera.intrinsic)
        msg.header.frame_id = self.name + "_optical_frame"
        msg.header.stamp = stamp
        self.info_pub.publish(msg)

        img = self.camera.get_image()
        msg = self.cv_bridge.cv2_to_imgmsg(img.depth)
        msg.header.stamp = stamp
        self.depth_pub.publish(msg)


def main():
    rospy.init_node("bt_sim")
    server = BtSimNode()
    server.run()


if __name__ == "__main__":
    main()
