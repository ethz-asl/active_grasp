#!/usr/bin/env python3

from actionlib import SimpleActionServer
import control_msgs.msg
from controller_manager_msgs.srv import *
import cv_bridge
from franka_gripper.msg import *
from geometry_msgs.msg import Twist
import numpy as np
import rospy
from sensor_msgs.msg import JointState, Image, CameraInfo
import skimage.transform
from threading import Thread

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
            MoveActionPlugin(self.sim.gripper),
            GraspActionPlugin(self.sim.gripper),
            GripperActionPlugin(),
            CameraPlugin(self.sim.camera),
        ]
        self.controllers = {
            "cartesian_velocity_controller": CartesianVelocityControllerPlugin(
                self.sim.arm, self.sim.model
            ),
            "position_joint_trajectory_controller": JointTrajectoryControllerPlugin(
                self.sim.arm
            ),
        }

    def start_plugins(self):
        for plugin in self.plugins + list(self.controllers.values()):
            plugin.thread.start()

    def activate_plugins(self):
        for plugin in self.plugins:
            plugin.activate()

    def deactivate_plugins(self):
        for plugin in self.plugins:
            plugin.deactivate()

    def deactivate_controllers(self):
        for controller in self.controllers.values():
            controller.deactivate()

    def advertise_services(self):
        rospy.Service("seed", Seed, self.seed)
        rospy.Service("reset", Reset, self.reset)
        rospy.Service(
            "/controller_manager/switch_controller",
            SwitchController,
            self.switch_controller,
        )

    def seed(self, req):
        self.sim.seed(req.seed)
        return SeedResponse()

    def reset(self, req):
        self.deactivate_plugins()
        self.deactivate_controllers()
        rospy.sleep(1.0)  # TODO replace with a read-write lock
        bbox = self.sim.reset()
        self.activate_plugins()
        return ResetResponse(to_bbox_msg(bbox))

    def switch_controller(self, req):
        for controller in req.stop_controllers:
            self.controllers[controller].deactivate()
        for controller in req.start_controllers:
            self.controllers[controller].activate()
        return SwitchControllerResponse(ok=True)

    def run(self):
        self.start_plugins()
        self.activate_plugins()
        rospy.spin()


class Plugin:
    """A plugin that spins at a constant rate in its own thread."""

    def __init__(self, rate):
        self.rate = rate
        self.thread = Thread(target=self.loop, daemon=True)
        self.is_running = False

    def activate(self):
        self.is_running = True

    def deactivate(self):
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


class CartesianVelocityControllerPlugin(Plugin):
    def __init__(self, arm, model, rate=30):
        super().__init__(rate)
        self.arm = arm
        self.model = model
        rospy.Subscriber("command", Twist, self.target_cb)

    def target_cb(self, msg):
        self.dx_d = from_twist_msg(msg)

    def activate(self):
        self.dx_d = np.zeros(6)
        self.is_running = True

    def deactivate(self):
        self.dx_d = np.zeros(6)
        self.is_running = False
        self.arm.set_desired_joint_velocities(np.zeros(7))

    def update(self):
        q, _ = self.arm.get_state()
        J_pinv = np.linalg.pinv(self.model.jacobian(q))
        cmd = np.dot(J_pinv, self.dx_d)
        self.arm.set_desired_joint_velocities(cmd)


class JointTrajectoryControllerPlugin(Plugin):
    def __init__(self, arm, rate=30):
        super().__init__(rate)
        self.arm = arm
        self.dt = 1.0 / self.rate
        self.init_action_server()

    def init_action_server(self):
        name = "position_joint_trajectory_controller/follow_joint_trajectory"
        self.action_server = SimpleActionServer(
            name,
            control_msgs.msg.FollowJointTrajectoryAction,
            auto_start=False,
        )
        self.action_server.register_goal_callback(self.action_goal_cb)
        self.action_server.start()

    def action_goal_cb(self):
        goal = self.action_server.accept_new_goal()
        self.elapsed_time = 0.0
        self.points = iter(goal.trajectory.points)
        self.next_point = next(self.points)

    def update(self):
        if self.action_server.is_active():
            self.elapsed_time += self.dt
            if self.elapsed_time > self.next_point.time_from_start.to_sec():
                try:
                    self.next_point = next(self.points)
                except StopIteration:
                    self.action_server.set_succeeded()
                    return
            self.arm.set_desired_joint_positions(self.next_point.positions)


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


class GripperActionPlugin(Plugin):
    """Empty action server to make MoveIt happy"""

    def __init__(self, rate=1):
        super().__init__(rate)
        self.init_action_server()

    def init_action_server(self):
        name = "/franka_gripper/gripper_action"
        self.action_server = SimpleActionServer(
            name, control_msgs.msg.GripperCommandAction, auto_start=False
        )
        self.action_server.register_goal_callback(self.action_goal_cb)
        self.action_server.start()

    def action_goal_cb(self):
        self.action_server.accept_new_goal()

    def update(self):
        if self.action_server.is_active():
            self.action_server.set_succeeded()


class CameraPlugin(Plugin):
    def __init__(self, camera, name="camera", rate=5):
        super().__init__(rate)
        self.camera = camera
        self.name = name
        self.cam_noise = rospy.get_param("~cam_noise", True)
        if rospy.get_param("~calib_error"):
            self.camera.calib_error = Transform(
                Rotation.from_euler("xyz", [0.27, 0.034, 0.18], degrees=True),
                np.r_[0.002, 0.0018, 0.0007],
            )
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

        if self.cam_noise:
            depth = apply_noise(depth)

        msg = self.cv_bridge.cv2_to_imgmsg(depth)
        msg.header.stamp = stamp
        self.depth_pub.publish(msg)


def apply_noise(img, k=1000, theta=0.001, sigma=0.005, l=4.0):
    # Multiplicative and additive noise
    img *= np.random.gamma(k, theta)
    h, w = img.shape
    noise = np.random.randn(int(h / l), int(w / l)) * sigma
    img += skimage.transform.resize(noise, img.shape, order=1, mode="constant")
    return img


def main():
    rospy.init_node("bt_sim")
    server = BtSimNode()
    server.run()


if __name__ == "__main__":
    main()
