import numpy as np

from .policy import SingleViewPolicy, MultiViewPolicy, compute_error


class InitialView(SingleViewPolicy):
    def update(self, img, pose):
        self.x_d = pose
        cmd = super().update(img, pose)
        return cmd


class TopView(SingleViewPolicy):
    def activate(self, bbox):
        super().activate(bbox)
        self.x_d = self.view_sphere.get_view(0.0, 0.0)
        self.done = False if self.is_view_feasible(self.x_d) else True


class TopTrajectory(MultiViewPolicy):
    def activate(self, bbox):
        super().activate(bbox)
        self.x_d = self.view_sphere.get_view(0.0, 0.0)
        self.done = False if self.is_view_feasible(self.x_d) else True

    def update(self, img, x):
        self.integrate(img, x)
        linear, angular = compute_error(self.x_d, x)
        if np.linalg.norm(linear) < 0.02:
            self.done = True
            return np.zeros(6)
        else:
            return self.compute_velocity_cmd(linear, angular)
