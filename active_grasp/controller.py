from geometry_msgs.msg import PoseStamped
import numpy as np
import rospy
import time

from active_grasp.bbox import from_bbox_msg
from active_grasp.srv import Reset, ResetRequest
from robot_helpers.ros.conversions import to_pose_stamped_msg
from robot_helpers.ros.panda import PandaGripperClient
from robot_helpers.spatial import Rotation, Transform


class GraspController:
    def __init__(self, policy):
        self.policy = policy
        self._reset_env = rospy.ServiceProxy("reset", Reset)
        self._load_parameters()
        self._init_robot_control()

    def _load_parameters(self):
        self.T_G_EE = Transform.from_list(rospy.get_param("~ee_grasp_offset")).inv()

    def _init_robot_control(self):
        self.target_pose_pub = rospy.Publisher("command", PoseStamped, queue_size=10)
        self.gripper = PandaGripperClient()

    def _send_cmd(self, pose):
        msg = to_pose_stamped_msg(pose, "panda_link0")
        self.target_pose_pub.publish(msg)

    def run(self):
        bbox = self._reset()
        with Timer("search_time"):
            grasp = self._search_grasp(bbox)
        res = self._execute_grasp(grasp)
        return self._collect_info(res)

    def _reset(self):
        res = self._reset_env(ResetRequest())
        rospy.sleep(1.0)  # wait for states to be updated
        return from_bbox_msg(res.bbox)

    def _search_grasp(self, bbox):
        self.policy.activate(bbox)
        r = rospy.Rate(self.policy.rate)
        while True:
            cmd = self.policy.update()
            if self.policy.done:
                break
            self._send_cmd(cmd)
            r.sleep()
        return self.policy.best_grasp

    def _execute_grasp(self, grasp):
        if not grasp:
            return "aborted"

        T_B_G = self._postprocess(grasp)
        self.gripper.move(0.08)

        # Move to an initial pose offset.
        self._send_cmd(T_B_G * Transform.translation([0, 0, -0.05]) * self.T_G_EE)
        rospy.sleep(3.0)

        # Approach grasp pose.
        self._send_cmd(T_B_G * self.T_G_EE)
        rospy.sleep(2.0)

        # Close the fingers.
        self.gripper.grasp()

        # Lift the object.
        target = Transform.translation([0, 0, 0.2]) * T_B_G * self.T_G_EE
        self._send_cmd(target)
        rospy.sleep(2.0)

        # Check whether the object remains in the hand
        success = self.gripper.read() > 0.005

        return "succeeded" if success else "failed"

    def _postprocess(self, T_B_G):
        # Ensure that the camera is pointing forward.
        rot = T_B_G.rotation
        if rot.as_matrix()[:, 0][0] < 0:
            T_B_G.rotation = rot * Rotation.from_euler("z", np.pi)
        return T_B_G

    def _collect_info(self, result):
        points = [p.translation for p in self.policy.viewpoints]
        d = np.sum([np.linalg.norm(p2 - p1) for p1, p2 in zip(points, points[1:])])
        info = {
            "result": result,
            "viewpoint_count": len(points),
            "distance": d,
        }
        info.update(Timer.timers)
        return info


class Timer:
    timers = dict()

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_info):
        self.stop()

    def start(self):
        self.tic = time.perf_counter()

    def stop(self):
        elapsed_time = time.perf_counter() - self.tic
        self.timers[self.name] = elapsed_time
