import numpy as np

from .policy import BasePolicy
from vgn.utils import look_at


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
        eye = np.r_[self.center[:2], self.center[2] + 0.3]
        up = np.r_[1.0, 0.0, 0.0]
        return [look_at(eye, self.center, up)]

    def compute_ig(self, view):
        return 1.0

    def compute_cost(self, view):
        return 1.0

    def check_done(self):
        return len(self.viewpoints) == 20
