import itertools
import numpy as np
import rospy

from .policy import BasePolicy
from vgn.utils import look_at, spherical_to_cartesian


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
        self.visualizer.views(self.base_frame, self.intrinsic, views, utilities)

        # Determine next-best-view
        nbv = views[np.argmax(utilities)]

        if self.check_done():
            self.best_grasp = self.compute_best_grasp()
            self.done = True
            return

        return nbv.inv()  # the controller expects T_cam_base

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

        t_min = 0.2
        t_max = corners[2].max()  # TODO This bound might be a bit too short
        t_step = 0.01

        view = self.T_task_base * view  # We'll work in the task frame from now on

        for u in range(u_min, u_max):
            for v in range(v_min, v_max):
                origin = view.translation
                direction = np.r_[(u - cx) / fx, (v - cy) / fy, 1.0]
                direction = view.rotation.apply(direction / np.linalg.norm(direction))

                self.visualizer.rays(self.task_frame, origin, [direction])
                rospy.sleep(0.1)

                t = t_min
                while t < t_max:
                    p = origin + t * direction
                    t += t_step

                    self.visualizer.point(self.task_frame, p)
                    rospy.sleep(0.1)

        return 1.0

    def compute_cost(self, view):
        return 1.0

    def check_done(self):
        return len(self.viewpoints) == 4
