import itertools
import numpy as np
from robot_helpers.perception import CameraIntrinsic
from robot_helpers.spatial import Transform
import rospy

from .policy import BasePolicy
from vgn.utils import look_at, spherical_to_cartesian


class Ray:
    def __init__(self, origin, direction):
        self.o = origin
        self.d = direction

    def __call__(self, t):
        return self.o + self.d * t


class NextBestView(BasePolicy):
    def __init__(self, rate, filter_grasps):
        super().__init__(rate, filter_grasps)

    def activate(self, bbox):
        super().activate(bbox)

    def update(self, img, extrinsic):
        # Integrate latest measurement
        self.integrate_img(img, extrinsic)

        # Generate viewpoints
        views = self.generate_viewpoints()

        # Evaluate viewpoints
        gains = [self.compute_ig(v) for v in views]
        costs = [self.compute_cost(v) for v in views]
        utilities = gains / np.sum(gains) - costs / np.sum(costs)

        # Visualize
        self.visualizer.views(self.intrinsic, views, utilities)

        # Determine next-best-view
        nbv = views[np.argmax(utilities)]

        if self.check_done():
            self.best_grasp = self.compute_best_grasp()
            self.done = True
        else:
            return nbv

    def generate_viewpoints(self):
        r, h = 0.14, 0.2
        thetas = np.arange(1, 4) * np.deg2rad(30)
        phis = np.arange(1, 6) * np.deg2rad(60)
        views = []
        for theta, phi in itertools.product(thetas, phis):
            eye = self.center + np.r_[0, 0, h] + spherical_to_cartesian(r, theta, phi)
            target = self.center
            up = np.r_[1.0, 0.0, 0.0]
            views.append(look_at(eye, target, up).inv())
        return views

    def compute_ig(self, view, downsample=20):
        fx = self.intrinsic.fx / downsample
        fy = self.intrinsic.fy / downsample
        cx = self.intrinsic.cx / downsample
        cy = self.intrinsic.cy / downsample

        T_cam_base = view.inv()
        corners = np.array([T_cam_base.apply(p) for p in self.bbox.corners]).T
        u = (fx * corners[0] / corners[2] + cx).round().astype(int)
        v = (fy * corners[1] / corners[2] + cy).round().astype(int)
        u_min, u_max = u.min(), u.max()
        v_min, v_max = v.min(), v.max()

        for u in range(u_min, u_max):
            for v in range(v_min, v_max):
                direction = np.r_[(u - cx) / fx, (v - cy) / fy, 1.0]
                direction = direction / np.linalg.norm(direction)
                direction = view.rotation.apply(direction)
                ray = Ray(view.translation, direction)

        return 1.0

    def compute_cost(self, view):
        return 1.0

    def check_done(self):
        return len(self.viewpoints) == 20
