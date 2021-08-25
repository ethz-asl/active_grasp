import numpy as np


from .policy import SingleViewPolicy
from vgn.utils import look_at


class InitialView(SingleViewPolicy):
    def update(self, img, extrinsic):
        self.target = extrinsic
        super().update(img, extrinsic)


class FrontView(SingleViewPolicy):
    def activate(self, bbox):
        super().activate(bbox)
        l, theta = 0.25, np.deg2rad(30)
        eye = np.r_[
            self.center[0] - l * np.sin(theta),
            self.center[1],
            self.center[2] + l * np.cos(theta),
        ]
        up = np.r_[1.0, 0.0, 0.0]
        self.target = look_at(eye, self.center, up)


class TopView(SingleViewPolicy):
    def activate(self, bbox):
        super().activate(bbox)
        eye = np.r_[self.center[:2], self.center[2] + 0.25]
        up = np.r_[1.0, 0.0, 0.0]
        self.target = look_at(eye, self.center, up)
