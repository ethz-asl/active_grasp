import numpy as np
from sensor_msgs.msg import CameraInfo
from pathlib import Path
import rospy

from .visualization import Visualizer
from robot_helpers.model import KDLModel
from robot_helpers.ros import tf
from robot_helpers.ros.conversions import *
from vgn.detection import *
from vgn.perception import UniformTSDFVolume


class Policy:
    def __init__(self):
        self.load_parameters()
        self.init_robot_model()
        self.init_visualizer()

    def load_parameters(self):
        self.base_frame = rospy.get_param("~base_frame_id")
        self.cam_frame = rospy.get_param("~camera/frame_id")
        self.task_frame = "task"
        info_topic = rospy.get_param("~camera/info_topic")
        msg = rospy.wait_for_message(info_topic, CameraInfo, rospy.Duration(2.0))
        self.intrinsic = from_camera_info_msg(msg)
        self.qual_threshold = rospy.get_param("vgn/qual_threshold")

    def init_robot_model(self):
        self.model = KDLModel.from_parameter_server(self.base_frame, self.cam_frame)

    def init_visualizer(self):
        self.vis = Visualizer()

    def is_feasible(self, view, q_init=None):
        q_init = q_init if q_init else [0.0, -0.79, 0.0, -2.356, 0.0, 1.57, 0.79]
        return self.model.ik(q_init, view) is not None

    def activate(self, bbox, view_sphere):
        self.vis.clear()

        self.bbox = bbox
        self.view_sphere = view_sphere
        self.vis.bbox(self.base_frame, self.bbox)

        self.calibrate_task_frame()

        self.tsdf = UniformTSDFVolume(0.3, 40)
        self.vgn = VGN(Path(rospy.get_param("vgn/model")))

        self.views = []
        self.best_grasp = None
        self.x_d = None
        self.done = False

    def calibrate_task_frame(self):
        self.T_base_task = Transform.translation(self.bbox.center - np.full(3, 0.15))
        self.T_task_base = self.T_base_task.inv()
        tf.broadcast(self.T_base_task, self.base_frame, self.task_frame)
        rospy.sleep(0.5)  # Wait for tf tree to be updated
        self.vis.workspace(self.task_frame, 0.3)

    def update(self, img, pose):
        raise NotImplementedError

    def sort_grasps(self, in_grasps):
        # Transforms grasps into base frame, checks whether they lie on the target, and sorts by their score
        grasps, scores = [], []

        for grasp in in_grasps:
            pose = self.T_base_task * grasp.pose
            R, t = pose.rotation, pose.translation

            # Filter out artifacts close to the support
            if t[2] < self.bbox.min[2] + 0.04:
                continue

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


class SingleViewPolicy(Policy):
    def update(self, img, x):
        linear, _ = compute_error(self.x_d, x)
        if np.linalg.norm(linear) < 0.02:
            self.views.append(x)
            self.tsdf.integrate(img, self.intrinsic, x.inv() * self.T_base_task)
            tsdf_grid, voxel_size = self.tsdf.get_grid(), self.tsdf.voxel_size
            self.vis.scene_cloud(self.task_frame, self.tsdf.get_scene_cloud())
            self.vis.map_cloud(self.task_frame, self.tsdf.get_map_cloud())

            out = self.vgn.predict(tsdf_grid)
            self.vis.quality(self.task_frame, voxel_size, out.qual, 0.5)

            grasps = select_grid(voxel_size, out, threshold=self.qual_threshold)
            grasps, scores = self.sort_grasps(grasps)
            self.vis.grasps(self.base_frame, grasps, scores)

            self.best_grasp = grasps[0] if len(grasps) > 0 else None
            self.done = True


class MultiViewPolicy(Policy):
    def activate(self, bbox, view_sphere):
        super().activate(bbox, view_sphere)
        self.T = 5  # Window size of grasp prediction history
        self.qual_hist = np.zeros((self.T,) + (40,) * 3, np.float32)

    def integrate(self, img, x):
        self.views.append(x)
        self.tsdf.integrate(img, self.intrinsic, x.inv() * self.T_base_task)

        self.vis.scene_cloud(self.task_frame, self.tsdf.get_scene_cloud())
        self.vis.map_cloud(self.task_frame, self.tsdf.get_map_cloud())
        self.vis.path(self.base_frame, self.views)

        tsdf_grid, voxel_size = self.tsdf.get_grid(), self.tsdf.voxel_size
        out = self.vgn.predict(tsdf_grid)
        self.vis.quality(self.task_frame, self.tsdf.voxel_size, out.qual, 0.5)

        t = (len(self.views) - 1) % self.T
        self.qual_hist[t, ...] = out.qual

        grasps = select_grid(voxel_size, out, threshold=self.qual_threshold)
        grasps, scores = self.sort_grasps(grasps)

        if len(grasps) > 0:
            self.best_grasp = grasps[0]
            self.vis.best_grasp(self.base_frame, grasps[0], scores[0])

        self.vis.grasps(self.base_frame, grasps, scores)


def compute_error(x_d, x):
    linear = x_d.translation - x.translation
    angular = (x_d.rotation * x.rotation.inv()).as_rotvec()
    return linear, angular


registry = {}


def register(id, cls):
    global registry
    registry[id] = cls


def make(id, *args, **kwargs):
    if id in registry:
        return registry[id](*args, **kwargs)
    else:
        raise ValueError("{} policy does not exist.".format(id))
