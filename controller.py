import numpy as np
import rospy

from robot_utils.ros import tf
from robot_utils.ros.panda import PandaGripperClient
from robot_utils.spatial import Rotation, Transform
from utils import CartesianPoseControllerClient


class GraspController:
    def __init__(self, policy):
        self.policy = policy
        self.controller = CartesianPoseControllerClient()
        self.gripper = PandaGripperClient()
        self.load_parameters()

    def load_parameters(self):
        self.T_G_EE = Transform.from_list(rospy.get_param("~ee_grasp_offset")).inv()

    def run(self):
        self.reset()
        grasp = self.explore()
        if grasp:
            self.execute_grasp(grasp)

    def reset(self):
        raise NotImplementedError

    def explore(self):
        self.policy.activate()
        r = rospy.Rate(self.policy.rate)
        while True:
            cmd = self.policy.update()
            if self.policy.done:
                break
            self.controller.send_target(cmd)
            r.sleep()
        return self.policy.best_grasp

    def execute_grasp(self, grasp):
        T_B_G = self.postprocess(grasp)

        self.gripper.move(0.08)

        # Move to an initial pose offset.
        self.controller.send_target(
            T_B_G * Transform.translation([0, 0, -0.05]) * self.T_G_EE
        )
        rospy.sleep(3.0)

        # Approach grasp pose.
        self.controller.send_target(T_B_G * self.T_G_EE)
        rospy.sleep(1.0)

        # Close the fingers.
        self.gripper.grasp()

        # Lift the object.
        target = Transform.translation([0, 0, 0.2]) * T_B_G * self.T_G_EE
        self.controller.send_target(target)
        rospy.sleep(2.0)

        # Check whether the object remains in the hand
        return self.gripper.read() > 0.005

    def postprocess(self, T_B_G):
        # Ensure that the camera is pointing forward.
        rot = T_B_G.rotation
        if rot.as_matrix()[:, 0][0] < 0:
            T_B_G.rotation = rot * Rotation.from_euler("z", np.pi)
        return T_B_G