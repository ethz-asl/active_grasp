import numpy as np

from .policy import SingleViewPolicy, MultiViewPolicy
from vgn.utils import look_at


class InitialView(SingleViewPolicy):
    def update(self, img, pose):
        self.x_d = pose
        cmd = super().update(img, pose)
        return cmd


class TopView(SingleViewPolicy):
    def activate(self, bbox):
        super().activate(bbox)
        eye = np.r_[self.center[:2], self.bbox.max[2] + self.min_z_dist]
        up = np.r_[1.0, 0.0, 0.0]
        self.x_d = look_at(eye, self.center, up)
        self.done = False if self.is_view_feasible(self.x_d) else True


class TopTrajectory(MultiViewPolicy):
    def activate(self, bbox):
        super().activate(bbox)
        eye = np.r_[self.center[:2], self.bbox.max[2] + self.min_z_dist]
        up = np.r_[1.0, 0.0, 0.0]
        self.x_d = look_at(eye, self.center, up)
        self.done = False if self.is_view_feasible(self.x_d) else True

    def update(self, img, x):
        self.integrate(img, x)
        linear, angular = self.compute_error(self.x_d, x)
        if np.linalg.norm(linear) < 0.02:
            self.done = True
            return np.zeros(6)
        else:
            return self.compute_velocity_cmd(linear, angular)
