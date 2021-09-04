import numpy as np
from sensor_msgs.msg import CameraInfo
from pathlib import Path
import rospy

from .visualization import Visualizer
from robot_helpers.ros import tf
from robot_helpers.ros.conversions import *
from vgn.detection import *
from vgn.perception import UniformTSDFVolume
from vgn.utils import *


class Policy:
    def __init__(self, rate=5):
        self.rate = rate
        self.load_parameters()
        self.init_visualizer()

    def load_parameters(self):
        self.base_frame = rospy.get_param("~base_frame_id")
        info_topic = rospy.get_param("~camera/info_topic")
        self.linear_vel = rospy.get_param("~linear_vel")
        self.min_z_dist = rospy.get_param("~camera/min_z_dist")
        self.qual_threshold = rospy.get_param("~qual_threshold")
        self.task_frame = "task"

        msg = rospy.wait_for_message(info_topic, CameraInfo, rospy.Duration(2.0))
        self.intrinsic = from_camera_info_msg(msg)

    def init_visualizer(self):
        self.vis = Visualizer()

    def activate(self, bbox):
        self.bbox = bbox
        self.calibrate_task_frame()
        self.vis.clear()
        self.vis.bbox(self.base_frame, bbox)
        self.tsdf = UniformTSDFVolume(0.3, 40)
        self.vgn = VGN(Path(rospy.get_param("vgn/model")))
        self.views = []
        self.best_grasp = None
        self.done = False

    def calibrate_task_frame(self):
        self.center = 0.5 * (self.bbox.min + self.bbox.max)
        self.T_base_task = Transform.translation(self.center - np.full(3, 0.15))
        self.T_task_base = self.T_base_task.inv()
        tf.broadcast(self.T_base_task, self.base_frame, self.task_frame)
        rospy.sleep(0.1)

    def compute_error(self, x_d, x):
        linear = x_d.translation - x.translation
        angular = (x_d.rotation * x.rotation.inv()).as_rotvec()
        return linear, angular

    def compute_velocity_cmd(self, linear, angular):
        kp = 4.0
        linear = kp * linear
        scale = np.linalg.norm(linear)
        linear *= np.clip(scale, 0.0, self.linear_vel) / scale
        return np.r_[linear, angular]

    def sort_grasps(self, in_grasps):
        # Transforms grasps into base frame, checks whether they lie on the target, and sorts by their score
        grasps, scores = [], []

        for grasp in in_grasps:
            pose = self.T_base_task * grasp.pose
            tip = pose.rotation.apply([0, 0, 0.05]) + pose.translation
            if self.bbox.is_inside(tip):
                grasp.pose = pose
                grasps.append(grasp)
                scores.append(self.score_fn(grasp))

        grasps, scores = np.asarray(grasps), np.asarray(scores)
        indices = np.argsort(-scores)
        return grasps[indices], scores[indices]

    def score_fn(self, grasp):
        return grasp.quality

    def update(self, img, pose):
        raise NotImplementedError


class SingleViewPolicy(Policy):
    def update(self, img, x):
        linear, angular = self.compute_error(self.x_d, x)

        if np.linalg.norm(linear) < 0.02:
            self.views.append(x)

            self.tsdf.integrate(img, self.intrinsic, x.inv() * self.T_base_task)
            tsdf_grid, voxel_size = self.tsdf.get_grid(), self.tsdf.voxel_size
            self.vis.scene_cloud(self.task_frame, self.tsdf.get_scene_cloud())
            self.vis.map_cloud(self.task_frame, self.tsdf.get_map_cloud())

            out = self.vgn.predict(tsdf_grid)
            self.vis.quality(self.task_frame, voxel_size, out.qual)

            grasps = select_grid(voxel_size, out, threshold=self.qual_threshold)
            grasps, scores = self.sort_grasps(grasps)

            self.vis.grasps(self.base_frame, grasps, scores)

            self.best_grasp = grasps[0] if len(grasps) > 0 else None
            self.done = True
            return np.zeros(6)
        else:
            return self.compute_velocity_cmd(linear, angular)


class MultiViewPolicy(Policy):
    def integrate(self, img, x):
        self.views.append(x)
        self.tsdf.integrate(img, self.intrinsic, x.inv() * self.T_base_task)

        self.vis.scene_cloud(self.task_frame, self.tsdf.get_scene_cloud())
        self.vis.map_cloud(self.task_frame, self.tsdf.get_map_cloud())
        self.vis.path(self.base_frame, self.views)

        tsdf_grid, voxel_size = self.tsdf.get_grid(), self.tsdf.voxel_size
        out = self.vgn.predict(tsdf_grid)

        grasps = select_grid(voxel_size, out, threshold=self.qual_threshold)
        grasps, scores = self.sort_grasps(grasps)

        if len(grasps) > 0:
            self.best_grasp = grasps[0]
            self.vis.best_grasp(self.base_frame, grasps[0], scores[0])

        self.vis.grasps(self.base_frame, grasps, scores)


registry = {}


def register(id, cls):
    global registry
    registry[id] = cls


def make(id, *args, **kwargs):
    if id in registry:
        return registry[id](*args, **kwargs)
    else:
        raise ValueError("{} policy does not exist.".format(id))
