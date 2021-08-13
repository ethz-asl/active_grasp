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


    def compute_ig(self, view):
        return 1.0

    def compute_cost(self, view):
        return 1.0

    def check_done(self):
        return len(self.viewpoints) == 20
