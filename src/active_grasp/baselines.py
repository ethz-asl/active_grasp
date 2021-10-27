import numpy as np

from .policy import SingleViewPolicy, MultiViewPolicy, compute_error


class InitialView(SingleViewPolicy):
    def update(self, img, x, q):
        self.x_d = x
        super().update(img, x, q)


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

    def update(self, img, x, q):
        self.integrate(img, x, q)
        linear, _ = compute_error(self.x_d, x)
        if np.linalg.norm(linear) < 0.02:
            self.done = True


class FixedTrajectory(MultiViewPolicy):
    def activate(self, bbox, view_sphere):
        pass

    def update(self, img, x, q):
        pass
