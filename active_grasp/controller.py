from controller_manager_msgs.srv import *
import copy
import cv_bridge
from geometry_msgs.msg import Twist
import numpy as np
import rospy
from sensor_msgs.msg import Image

from .bbox import from_bbox_msg
from .timer import Timer
from active_grasp.srv import Reset, ResetRequest
from robot_helpers.ros import tf
from robot_helpers.ros.conversions import *
from robot_helpers.ros.panda import PandaGripperClient
from robot_helpers.ros.moveit import MoveItClient
from robot_helpers.spatial import Rotation, Transform


class GraspController:
    def __init__(self, policy):
        self.policy = policy
        self.load_parameters()
        self.lookup_transforms()
        self.init_service_proxies()
        self.init_robot_connection()
        self.init_camera_stream()

    def load_parameters(self):
        self.base_frame = rospy.get_param("~base_frame_id")
        self.ee_frame = rospy.get_param("~ee_frame_id")
        self.cam_frame = rospy.get_param("~camera/frame_id")
        self.depth_topic = rospy.get_param("~camera/depth_topic")
        self.T_grasp_ee = Transform.from_list(rospy.get_param("~ee_grasp_offset")).inv()

    def lookup_transforms(self):
        tf.init()
        self.T_ee_cam = tf.lookup(self.ee_frame, self.cam_frame)

    def init_service_proxies(self):
        self.reset_env = rospy.ServiceProxy("reset", Reset)
        self.switch_controller = rospy.ServiceProxy(
            "controller_manager/switch_controller", SwitchController
        )

    def init_robot_connection(self):
        self.cartesian_vel_pub = rospy.Publisher("command", Twist, queue_size=10)
        self.gripper = PandaGripperClient()
        self.moveit = MoveItClient("panda_arm")

    def switch_to_cartesian_velocity_control(self):
        req = SwitchControllerRequest()
        req.start_controllers = ["cartesian_velocity_controller"]
        req.stop_controllers = ["position_joint_trajectory_controller"]
        self.switch_controller(req)

    def switch_to_joint_trajectory_control(self):
        req = SwitchControllerRequest()
        req.start_controllers = ["position_joint_trajectory_controller"]
        req.stop_controllers = ["cartesian_velocity_controller"]
        self.switch_controller(req)

    def init_camera_stream(self):
        self.cv_bridge = cv_bridge.CvBridge()
        rospy.Subscriber(self.depth_topic, Image, self.sensor_cb, queue_size=1)

    def sensor_cb(self, msg):
        self.latest_depth_msg = msg

    def run(self):
        bbox = self.reset()

        self.switch_to_cartesian_velocity_control()
        with Timer("search_time"):
            grasp = self.search_grasp(bbox)

        self.switch_to_joint_trajectory_control()
        with Timer("execution_time"):
            res = self.execute_grasp(grasp)

        return self.collect_info(res)

    def reset(self):
        res = self.reset_env(ResetRequest())
        rospy.sleep(1.0)  # wait for states to be updated
        return from_bbox_msg(res.bbox)

    def search_grasp(self, bbox):
        self.policy.activate(bbox)
        r = rospy.Rate(self.policy.rate)
        while not self.policy.done:
            img, pose = self.get_state()
            cmd = self.policy.update(img, pose)
            self.cartesian_vel_pub.publish(to_twist_msg(cmd))
            r.sleep()
        return self.policy.best_grasp

    def get_state(self):
        msg = copy.deepcopy(self.latest_depth_msg)
        img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32)
        pose = tf.lookup(self.base_frame, self.cam_frame, msg.header.stamp)
        return img, pose

    def execute_grasp(self, grasp):
        if not grasp:
            return "aborted"

        T_base_grasp = self.postprocess(grasp.pose)

        self.gripper.move(0.08)

        self.moveit.goto(T_base_grasp * Transform.t([0, 0, -0.05]) * self.T_grasp_ee)
        self.moveit.goto(T_base_grasp * self.T_grasp_ee)
        self.gripper.grasp()
        self.moveit.goto(Transform.t([0, 0, 0.1]) * T_base_grasp * self.T_grasp_ee)

        success = self.gripper.read() > 0.005

        return "succeeded" if success else "failed"

    def postprocess(self, T_base_grasp):
        rot = T_base_grasp.rotation
        if rot.as_matrix()[:, 0][0] < 0:  # Ensure that the camera is pointing forward
            T_base_grasp.rotation = rot * Rotation.from_euler("z", np.pi)
        return T_base_grasp

    def collect_info(self, result):
        points = [p.translation for p in self.policy.views]
        d = np.sum([np.linalg.norm(p2 - p1) for p1, p2 in zip(points, points[1:])])
        info = {
            "result": result,
            "view_count": len(points),
            "distance": d,
        }
        info.update(Timer.timers)
        return info
