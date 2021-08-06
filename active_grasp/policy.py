import numpy as np
from sensor_msgs.msg import CameraInfo
from pathlib import Path
import rospy

from .visualization import Visualizer
from robot_helpers.ros import tf
from robot_helpers.ros.conversions import *
from vgn.detection import VGN, compute_grasps
from vgn.perception import UniformTSDFVolume
from vgn.utils import *


class Policy:
    def activate(self, bbox):
        raise NotImplementedError

    def update(self, img, extrinsic):
        raise NotImplementedError


class BasePolicy(Policy):
    def __init__(self, rate=5):
        self.rate = rate
        self.load_parameters()
        self.init_visualizer()

    def load_parameters(self):
        self.base_frame = rospy.get_param("active_grasp/base_frame_id")
        self.task_frame = "task"
        info_topic = rospy.get_param("active_grasp/camera/info_topic")
        msg = rospy.wait_for_message(info_topic, CameraInfo, rospy.Duration(2.0))
        self.intrinsic = from_camera_info_msg(msg)
        self.vgn = VGN(Path(rospy.get_param("vgn/model")))

    def init_visualizer(self):
        self.visualizer = Visualizer(self.base_frame)

    def activate(self, bbox):
        self.bbox = bbox

        # Define the VGN task frame s.t. the bounding box is in its center
        self.center = 0.5 * (bbox.min + bbox.max)
        self.T_base_task = Transform.translation(self.center - np.full(3, 0.15))
        tf.broadcast(self.T_base_task, self.base_frame, self.task_frame)
        rospy.sleep(0.1)  # wait for the transform to be published

        self.tsdf = UniformTSDFVolume(0.3, 40)
        self.viewpoints = []
        self.done = False
        self.best_grasp = None

        self.visualizer.clear()
        self.visualizer.bbox(bbox)

    def integrate_img(self, img, extrinsic):
        self.viewpoints.append(extrinsic.inv())
        self.tsdf.integrate(img, self.intrinsic, extrinsic * self.T_base_task)
        self.visualizer.scene_cloud(self.task_frame, self.tsdf.get_scene_cloud())
        self.visualizer.path(self.viewpoints)

    def compute_best_grasp(self):
        return self.predict_best_grasp()

    def predict_best_grasp(self):
        tsdf_grid = self.tsdf.get_grid()
        out = self.vgn.predict(tsdf_grid)
        score_fn = lambda g: g.pose.translation[2]
        grasps = compute_grasps(self.tsdf.voxel_size, out, score_fn, max_filter_size=3)
        grasps = self.transform_grasps_to_base_frame(grasps)
        grasps = self.select_grasps_on_target_object(grasps)
        return grasps[0] if len(grasps) > 0 else None

    def transform_grasps_to_base_frame(self, grasps):
        for grasp in grasps:
            grasp.pose = self.T_base_task * grasp.pose
        return grasps

    def select_grasps_on_target_object(self, grasps):
        result = []
        for grasp in grasps:
            tip = grasp.pose.rotation.apply([0, 0, 0.05]) + grasp.pose.translation
            if self.bbox.is_inside(tip):
                result.append(grasp)
        return result


registry = {}


def register(id, cls):
    global registry
    registry[id] = cls


def make(id, *args, **kwargs):
    if id in registry:
        return registry[id](*args, **kwargs)
    else:
        raise ValueError("{} policy does not exist.".format(id))
