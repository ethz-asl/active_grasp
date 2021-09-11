import numpy as np

from .policy import SingleViewPolicy, MultiViewPolicy, compute_error


class InitialView(SingleViewPolicy):
    def update(self, img, pose):
        self.x_d = pose
        super().update(img, pose)


class TopView(SingleViewPolicy):
    def activate(self, bbox, view_sphere):
        super().activate(bbox, view_sphere)
        self.x_d = self.view_sphere.get_view(0.0, 0.0)
        self.done = False if self.is_feasible(self.x_d) else True


class TopTrajectory(MultiViewPolicy):
    def activate(self, bbox, view_sphere):
        super().activate(bbox, view_sphere)
        self.x_d = self.view_sphere.get_view(0.0, 0.0)
        self.done = False if self.is_feasible(self.x_d) else True

    def update(self, img, x):
        self.integrate(img, x)
        linear, _ = compute_error(self.x_d, x)
        if np.linalg.norm(linear) < 0.02:
            self.done = True
