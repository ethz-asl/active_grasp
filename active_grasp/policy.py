import cv_bridge
import numpy as np
from pathlib import Path
import rospy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

from .visualization import Visualizer
from robot_helpers.ros import tf
from robot_helpers.ros.conversions import *
from vgn.detection import VGN, compute_grasps
from vgn.perception import UniformTSDFVolume
from vgn.utils import *


class Policy:
    def activate(self, bbox):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class BasePolicy(Policy):
    def __init__(self):
        self.cv_bridge = cv_bridge.CvBridge()
        self.vgn = VGN(Path(rospy.get_param("vgn/model")))
        self.finger_depth = 0.05
        self.rate = 5
        self._load_parameters()
        self._lookup_transforms()
        self._init_camera_stream()
        self._init_publishers()
        self._init_visualizer()

    def _load_parameters(self):
        self.task_frame = rospy.get_param("~frame_id")
        self.base_frame = rospy.get_param("~base_frame_id")
        self.ee_frame = rospy.get_param("~ee_frame_id")
        self.cam_frame = rospy.get_param("~camera/frame_id")
        self.info_topic = rospy.get_param("~camera/info_topic")
        self.depth_topic = rospy.get_param("~camera/depth_topic")

    def _lookup_transforms(self):
        self.T_B_task = tf.lookup(self.base_frame, self.task_frame)
        self.T_EE_cam = tf.lookup(self.ee_frame, self.cam_frame)

    def _init_camera_stream(self):
        msg = rospy.wait_for_message(self.info_topic, CameraInfo, rospy.Duration(2.0))
        self.intrinsic = from_camera_info_msg(msg)
        rospy.Subscriber(self.depth_topic, Image, self._sensor_cb, queue_size=1)

    def _sensor_cb(self, msg):
        self.img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32)
        self.extrinsic = tf.lookup(self.cam_frame, self.task_frame, msg.header.stamp)

    def _init_publishers(self):
        self.scene_cloud_pub = rospy.Publisher("scene_cloud", PointCloud2, queue_size=1)

    def _init_visualizer(self):
        self.visualizer = Visualizer(self.task_frame)

    def activate(self, bbox):
        self.bbox = bbox
        self.tsdf = UniformTSDFVolume(0.3, 40)
        self.viewpoints = []
        self.done = False
        self.best_grasp = None  # grasp pose defined w.r.t. the robot's base frame
        self.visualizer.clear()
        self.visualizer.bbox(bbox)

    def _integrate_latest_image(self):
        self.viewpoints.append(self.extrinsic.inv())
        self.tsdf.integrate(
            self.img,
            self.intrinsic,
            self.extrinsic,
        )
        self._publish_scene_cloud()
        self.visualizer.path(self.viewpoints)

    def _publish_scene_cloud(self):
        cloud = self.tsdf.get_scene_cloud()
        msg = to_cloud_msg(self.task_frame, np.asarray(cloud.points))
        self.scene_cloud_pub.publish(msg)

    def _predict_best_grasp(self):
        tsdf_grid = self.tsdf.get_grid()
        out = self.vgn.predict(tsdf_grid)
        score_fn = lambda g: g.pose.translation[2]
        grasps = compute_grasps(self.tsdf.voxel_size, out, score_fn, max_filter_size=3)
        grasps = self._select_grasps_on_target_object(grasps)
        return self.T_B_task * grasps[0].pose if len(grasps) > 0 else None

    def _select_grasps_on_target_object(self, grasps):
        result = []
        for g in grasps:
            tip = g.pose.rotation.apply([0, 0, 0.05]) + g.pose.translation
            if self.bbox.is_inside(tip):
                result.append(g)
        return result


registry = {}


def register(id, cls):
    global registry
    registry[id] = cls


def make(id):
    if id in registry:
        return registry[id]()
    else:
        raise ValueError("{} policy does not exist.".format(id))
