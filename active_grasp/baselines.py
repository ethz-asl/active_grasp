import numpy as np
import rospy
import scipy.interpolate

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
        eye = np.r_[self.center[:2], self.center[2] + 0.3]
        up = np.r_[1.0, 0.0, 0.0]
        self.x_d = look_at(eye, self.center, up)


class TopTrajectory(MultiViewPolicy):
    def activate(self, bbox):
        super().activate(bbox)
        eye = np.r_[self.center[:2], self.center[2] + 0.3]
        up = np.r_[1.0, 0.0, 0.0]
        self.x_d = look_at(eye, self.center, up)

    def update(self, img, x):
        self.integrate(img, x)
        linear, angular = self.compute_error(self.x_d, x)
        if np.linalg.norm(linear) < 0.02:
            self.done = True
            return np.zeros(6)
        else:
            return self.compute_velocity_cmd(linear, angular)


class CircularTrajectory(MultiViewPolicy):
    def __init__(self, rate):
        super().__init__(rate)
        self.r = 0.08
        self.h = 0.3
        self.duration = 2.0 * np.pi * self.r / self.linear_vel
        self.m = scipy.interpolate.interp1d([0.0, self.duration], [np.pi, 3.0 * np.pi])

    def activate(self, bbox):
        super().activate(bbox)
        self.tic = rospy.Time.now()

    def update(self, img, x):
        self.integrate(img, x)
        elapsed_time = (rospy.Time.now() - self.tic).to_sec()
        if elapsed_time > self.duration:
            self.done = True
            return np.zeros(6)
        else:
            t = self.m(elapsed_time)
            eye = self.center + np.r_[self.r * np.cos(t), self.r * np.sin(t), self.h]
            up = np.r_[1.0, 0.0, 0.0]
            x_d = look_at(eye, self.center, up)
            linear, angular = self.compute_error(x_d, x)
            return self.compute_velocity_cmd(linear, angular)
