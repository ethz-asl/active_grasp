import copy
import cv_bridge
from geometry_msgs.msg import PoseStamped
import numpy as np
import rospy
from sensor_msgs.msg import Image

from .bbox import from_bbox_msg
from .timer import Timer
from active_grasp.srv import Reset, ResetRequest
from robot_helpers.ros import tf
from robot_helpers.ros.conversions import *
from robot_helpers.ros.panda import PandaGripperClient
from robot_helpers.spatial import Rotation, Transform


class GraspController:
    def __init__(self, policy):
        self.policy = policy
        self.reset_env = rospy.ServiceProxy("reset", Reset)
        self.load_parameters()
        self.lookup_transforms()
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

    def init_robot_connection(self):
        self.target_pose_pub = rospy.Publisher("command", PoseStamped, queue_size=10)
        self.gripper = PandaGripperClient()

    def send_cmd(self, pose):
        msg = to_pose_stamped_msg(pose, self.base_frame)
        self.target_pose_pub.publish(msg)

    def init_camera_stream(self):
        self.cv_bridge = cv_bridge.CvBridge()
        rospy.Subscriber(self.depth_topic, Image, self.sensor_cb, queue_size=1)

    def sensor_cb(self, msg):
        self.latest_depth_msg = msg

    def run(self):
        bbox = self.reset()
        with Timer("search_time"):
            grasp = self.search_grasp(bbox)
        res = self.execute_grasp(grasp)
        return self.collect_info(res)

    def reset(self):
        res = self.reset_env(ResetRequest())
        rospy.sleep(1.0)  # wait for states to be updated
        return from_bbox_msg(res.bbox)

    def search_grasp(self, bbox):
        self.policy.activate(bbox)
        r = rospy.Rate(self.policy.rate)
        while True:
            img, extrinsic = self.get_state()
            next_extrinsic = self.policy.update(img, extrinsic)
            if self.policy.done:
                break
            self.send_cmd((self.T_ee_cam * next_extrinsic).inv())
            r.sleep()
        return self.policy.best_grasp

    def get_state(self):
        msg = copy.deepcopy(self.latest_depth_msg)
        img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32)
        extrinsic = tf.lookup(self.cam_frame, self.base_frame, msg.header.stamp)
        return img, extrinsic

    def execute_grasp(self, grasp):
        if not grasp:
            return "aborted"

        T_base_grasp = self.postprocess(grasp.pose)
        self.gripper.move(0.08)

        # Move to an initial pose offset.
        self.send_cmd(
            T_base_grasp * Transform.translation([0, 0, -0.05]) * self.T_grasp_ee
        )
        rospy.sleep(4.0)  # TODO

        # Approach grasp pose.
        self.send_cmd(T_base_grasp * self.T_grasp_ee)
        rospy.sleep(2.0)

        # Close the fingers.
        self.gripper.grasp()

        # Lift the object.
        target = Transform.translation([0, 0, 0.2]) * T_base_grasp * self.T_grasp_ee
        self.send_cmd(target)
        rospy.sleep(2.0)

        # Check whether the object remains in the hand
        success = self.gripper.read() > 0.005

        return "succeeded" if success else "failed"

    def postprocess(self, T_base_grasp):
        # Ensure that the camera is pointing forward.
        rot = T_base_grasp.rotation
        if rot.as_matrix()[:, 0][0] < 0:
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
