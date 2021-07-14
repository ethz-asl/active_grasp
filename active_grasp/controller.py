import numpy as np
import rospy

from active_grasp.srv import Reset, ResetRequest
from active_grasp.utils import *
from robot_utils.ros.panda import PandaGripperClient
from robot_utils.spatial import Rotation, Transform


class GraspController:
    def __init__(self, policy):
        self.policy = policy
        self.controller = CartesianPoseControllerClient()
        self.gripper = PandaGripperClient()
        self.reset_env = rospy.ServiceProxy("reset", Reset)
        self.load_parameters()

    def load_parameters(self):
        self.T_G_EE = Transform.from_list(rospy.get_param("~ee_grasp_offset")).inv()

    def run(self):
        bbox = self.reset()
        with Timer("exploration_time"):
            grasp = self.explore(bbox)
        with Timer("execution_time"):
            res = self.execute_grasp(grasp)
        return self.collect_info(res)

    def reset(self):
        req = ResetRequest()
        res = self.reset_env(req)
        rospy.sleep(1.0)  # wait for states to be updated
        return from_bbox_msg(res.bbox)

    def explore(self, bbox):
        self.policy.activate(bbox)
        r = rospy.Rate(self.policy.rate)
        while not self.policy.done:
            cmd = self.policy.update()
            if self.policy.done:
                break
            self.controller.send_target(cmd)
            r.sleep()
        return self.policy.best_grasp

    def execute_grasp(self, grasp):
        if not grasp:
            return "aborted"

        T_B_G = self.postprocess(grasp)

        self.gripper.move(0.08)

        # Move to an initial pose offset.
        self.controller.send_target(
            T_B_G * Transform.translation([0, 0, -0.05]) * self.T_G_EE
        )
        rospy.sleep(3.0)

        # Approach grasp pose.
        self.controller.send_target(T_B_G * self.T_G_EE)
        rospy.sleep(2.0)

        # Close the fingers.
        self.gripper.grasp()

        # Lift the object.
        target = Transform.translation([0, 0, 0.2]) * T_B_G * self.T_G_EE
        self.controller.send_target(target)
        rospy.sleep(2.0)

        # Check whether the object remains in the hand
        success = self.gripper.read() > 0.005

        return "succeeded" if success else "failed"

    def postprocess(self, T_B_G):
        # Ensure that the camera is pointing forward.
        rot = T_B_G.rotation
        if rot.as_matrix()[:, 0][0] < 0:
            T_B_G.rotation = rot * Rotation.from_euler("z", np.pi)
        return T_B_G

    def collect_info(self, result):
        points = [p.translation for p in self.policy.viewpoints]
        d = np.sum([np.linalg.norm(p2 - p1) for p1, p2 in zip(points, points[1:])])

        info = {
            "result": result,
            "viewpoint_count": len(points),
            "distance_travelled": d,
        }
        info.update(self.policy.info)
        info.update(Timer.timers)
        return info
