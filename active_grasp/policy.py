import numpy as np
from sensor_msgs.msg import CameraInfo
from pathlib import Path
import rospy
import warnings

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
        self.score_fn = lambda g: g.pose.translation[2]

    def init_visualizer(self):
        self.visualizer = Visualizer(self.base_frame)

    def activate(self, bbox):
        self.bbox = bbox

        # Define the VGN task frame s.t. the bounding box is in its center
        self.center = 0.5 * (bbox.min + bbox.max)
        self.T_base_task = Transform.translation(self.center - np.full(3, 0.15))
        tf.broadcast(self.T_base_task, self.base_frame, self.task_frame)
        rospy.sleep(0.1)  # wait for the transform to be published

        N, self.T = 40, 10  # spatial and temporal resolution
        grid_shape = (N,) * 3

        self.tsdf = UniformTSDFVolume(0.3, N)

        self.qual_hist = np.zeros((self.T,) + grid_shape, np.float32)
        self.rot_hist = np.zeros((self.T, 4) + grid_shape, np.float32)
        self.width_hist = np.zeros((self.T,) + grid_shape, np.float32)

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

        if self.filter_grasps:
            tsdf_grid = self.tsdf.get_grid()
            out = self.vgn.predict(tsdf_grid)
            t = (len(self.viewpoints) - 1) % self.T
            self.qual_hist[t, ...] = out.qual
            self.rot_hist[t, ...] = out.rot
            self.width_hist[t, ...] = out.width

    def compute_best_grasp(self):
        if self.filter_grasps:
            T = len(self.viewpoints) if len(self.viewpoints) // self.T == 0 else self.T
            mask = self.qual_hist[:T, ...] > 0.0
            # The next line prints a warning since some voxels have no grasp
            # predictions resulting in empty slices.
            qual = np.mean(self.qual_hist[:T, ...], axis=0, where=mask)
            qual = np.nan_to_num(qual, copy=False)
            qual = threshold_quality(qual, 0.9)
            index_list = select_local_maxima(qual, 3)

            grasps = []
            for (i, j, k) in index_list:
                ts = np.flatnonzero(self.qual_hist[:T, i, j, k])
                if len(ts) < 3:
                    continue
                oris = Rotation.from_quat([self.rot_hist[t, :, i, j, k] for t in ts])
                ori = oris.mean()
                # TODO check variance as well
                pos = np.array([i, j, k], dtype=np.float64)
                width = self.width_hist[ts, i, j, k].mean()
                quality = self.qual_hist[ts, i, j, k].mean()
                grasps.append(Grasp(Transform(ori, pos), width, quality))
        else:
            tsdf_grid = self.tsdf.get_grid()
            out = self.vgn.predict(tsdf_grid)
            qual = threshold_quality(out.qual, 0.9)
            index_list = select_local_maxima(qual, 3)
            grasps = [select_at(out, i) for i in index_list]

        grasps = [from_voxel_coordinates(g, self.tsdf.voxel_size) for g in grasps]
        grasps = self.transform_grasps_to_base_frame(grasps)
        grasps = self.select_grasps_on_target_object(grasps)
        grasps = sort_grasps(grasps, self.score_fn)

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
