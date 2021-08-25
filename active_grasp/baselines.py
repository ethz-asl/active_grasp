import numpy as np
import rospy
import scipy.interpolate

from .policy import SingleViewPolicy, MultiViewPolicy
from vgn.utils import look_at


class InitialView(SingleViewPolicy):
    def update(self, img, extrinsic):
        self.target = extrinsic
        super().update(img, extrinsic)


class TopView(SingleViewPolicy):
    def activate(self, bbox):
        super().activate(bbox)
        eye = np.r_[self.center[:2], self.center[2] + 0.3]
        up = np.r_[1.0, 0.0, 0.0]
        self.target = look_at(eye, self.center, up)


class TopTrajectory(MultiViewPolicy):
    def activate(self, bbox):
        super().activate(bbox)
        eye = np.r_[self.center[:2], self.center[2] + 0.3]
        up = np.r_[1.0, 0.0, 0.0]
        self.target = look_at(eye, self.center, up)

    def update(self, img, extrinsic):
        self.integrate(img, extrinsic)
        if np.linalg.norm(extrinsic.translation - self.target.translation) < 0.01:
            self.done = True
        else:
            return self.target


class CircularTrajectory(MultiViewPolicy):
    def __init__(self, rate):
        super().__init__(rate)
        self.r = 0.1
        self.h = 0.3
        self.duration = 12.0
        self.m = scipy.interpolate.interp1d([0, self.duration], [np.pi, 3.0 * np.pi])

    def activate(self, bbox):
        super().activate(bbox)
        self.tic = rospy.Time.now()

    def update(self, img, extrinsic):
        self.integrate(img, extrinsic)
        elapsed_time = (rospy.Time.now() - self.tic).to_sec()
        if elapsed_time > self.duration:
            self.done = True
        else:
            t = self.m(elapsed_time)
            eye = self.center + np.r_[self.r * np.cos(t), self.r * np.sin(t), self.h]
            up = np.r_[1.0, 0.0, 0.0]
            return look_at(eye, self.center, up)
