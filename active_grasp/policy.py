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
    def activate(self, bbox):
        raise NotImplementedError

    def update(self, img, extrinsic):
        raise NotImplementedError


class BasePolicy(Policy):
    def __init__(self, rate=5, filter_grasps=False):
        self.rate = rate
        self.filter_grasps = filter_grasps
        self.load_parameters()
        self.init_visualizer()

    def load_parameters(self):
        self.base_frame = rospy.get_param("active_grasp/base_frame_id")
        self.task_frame = "task"
        info_topic = rospy.get_param("active_grasp/camera/info_topic")
        msg = rospy.wait_for_message(info_topic, CameraInfo, rospy.Duration(2.0))
        self.intrinsic = from_camera_info_msg(msg)
        self.vgn = VGN(Path(rospy.get_param("vgn/model")))
        self.score_fn = lambda g: g.pose.translation[2]  # TODO

    def init_visualizer(self):
        self.visualizer = Visualizer()

    def activate(self, bbox):
        self.bbox = bbox

        self.center = 0.5 * (bbox.min + bbox.max)
        self.T_base_task = Transform.translation(self.center - np.full(3, 0.15))
        self.T_task_base = self.T_base_task.inv()
        tf.broadcast(self.T_base_task, self.base_frame, self.task_frame)
        rospy.sleep(1.0)  # wait for the transform to be published

        N, self.T = 40, 10
        grid_shape = (N,) * 3

        self.tsdf = UniformTSDFVolume(0.3, N)

        self.qual_hist = np.zeros((self.T,) + grid_shape, np.float32)
        self.rot_hist = np.zeros((self.T, 4) + grid_shape, np.float32)
        self.width_hist = np.zeros((self.T,) + grid_shape, np.float32)

        self.viewpoints = []
        self.done = False
        self.best_grasp = None

        self.visualizer.clear()
        self.visualizer.bbox(self.base_frame, bbox)

    def integrate_img(self, img, extrinsic):
        self.viewpoints.append(extrinsic.inv())
        self.tsdf.integrate(img, self.intrinsic, extrinsic * self.T_base_task)
        tsdf_grid, voxel_size = self.tsdf.get_grid(), self.tsdf.voxel_size

        if self.filter_grasps:
            out = self.vgn.predict(self.tsdf.get_grid())
            t = (len(self.viewpoints) - 1) % self.T
            self.qual_hist[t, ...] = out.qual
            self.rot_hist[t, ...] = out.rot
            self.width_hist[t, ...] = out.width

            mean_qual = self.compute_mean_quality()
            self.visualizer.quality(self.task_frame, voxel_size, mean_qual)

        self.visualizer.scene_cloud(self.task_frame, self.tsdf.get_scene_cloud())
        self.visualizer.map_cloud(self.task_frame, voxel_size, tsdf_grid)
        self.visualizer.path(self.base_frame, self.viewpoints)

    def compute_best_grasp(self):
        if self.filter_grasps:
            qual = self.compute_mean_quality()
            index_list = select_local_maxima(qual, 0.9, 3)
            grasps = [g for i in index_list if (g := self.select_mean_at(i))]
        else:
            out = self.vgn.predict(self.tsdf.get_grid())
            qual = out.qual
            index_list = select_local_maxima(qual, 0.9, 3)
            grasps = [select_at(out, i) for i in index_list]

        grasps = [from_voxel_coordinates(g, self.tsdf.voxel_size) for g in grasps]
        grasps = self.transform_and_reject(grasps)
        grasps = sort_grasps(grasps, self.score_fn)

        self.visualizer.quality(self.task_frame, self.tsdf.voxel_size, qual)
        self.visualizer.grasps(self.base_frame, grasps)

        return grasps[0] if len(grasps) > 0 else None

    def compute_mean_quality(self):
        qual = np.mean(self.qual_hist, axis=0, where=self.qual_hist > 0.0)
        return np.nan_to_num(qual, copy=False)  # mean of empty slices returns nan

    def select_mean_at(self, index):
        i, j, k = index
        ts = np.flatnonzero(self.qual_hist[:, i, j, k])
        if len(ts) < 3:
            return
        ori = Rotation.from_quat([self.rot_hist[t, :, i, j, k] for t in ts])
        pos = np.array([i, j, k], dtype=np.float64)
        width = self.width_hist[ts, i, j, k].mean()
        qual = self.qual_hist[ts, i, j, k].mean()
        return Grasp(Transform(ori.mean(), pos), width, qual)

    def transform_and_reject(self, grasps):
        result = []
        for grasp in grasps:
            pose = self.T_base_task * grasp.pose
            tip = pose.rotation.apply([0, 0, 0.05]) + pose.translation
            if self.bbox.is_inside(tip):
                grasp.pose = pose
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
