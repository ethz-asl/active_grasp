import numpy as np
import scipy.interpolate
import rospy

from active_grasp.policy import BasePolicy
from robot_helpers.ros import tf
from vgn.utils import look_at


class SingleView(BasePolicy):
    """
    Process a single image from the initial viewpoint.
    """

    def update(self):
        self._integrate_latest_image()
        self.best_grasp = self._predict_best_grasp()
        self.done = True


class TopView(BasePolicy):
    """
    Move the camera to a top-down view of the target object.
    """

    def activate(self, bbox):
        super().activate(bbox)
        center = (bbox.min + bbox.max) / 2.0
        eye = np.r_[center[:2], center[2] + 0.3]
        up = np.r_[1.0, 0.0, 0.0]
        self.target = self.T_B_task * (self.T_EE_cam * look_at(eye, center, up)).inv()

    def update(self):
        current = tf.lookup(self.base_frame, self.ee_frame)
        error = current.translation - self.target.translation

        if np.linalg.norm(error) < 0.01:
            self.best_grasp = self._predict_best_grasp()
            self.done = True
        else:
            self._integrate_latest_image()
            return self.target


class RandomView(BasePolicy):
    """
    Move the camera to a random viewpoint on a circle centered above the target.
    """

    def __init__(self):
        super().__init__()
        self.r = 0.06
        self.h = 0.3

    def activate(self, bbox):
        super().activate(bbox)
        circle_center = (bbox.min + bbox.max) / 2.0
        circle_center[2] += self.h
        t = np.random.uniform(np.pi, 3.0 * np.pi)
        eye = circle_center + np.r_[self.r * np.cos(t), self.r * np.sin(t), 0]
        center = (self.bbox.min + self.bbox.max) / 2.0
        up = np.r_[1.0, 0.0, 0.0]
        self.target = self.T_B_task * (self.T_EE_cam * look_at(eye, center, up)).inv()

    def update(self):
        current = tf.lookup(self.base_frame, self.ee_frame)
        error = current.translation - self.target.translation

        if np.linalg.norm(error) < 0.01:
            self.best_grasp = self._predict_best_grasp()
            self.done = True
        else:
            self._integrate_latest_image()
            return self.target


class FixedTrajectory(BasePolicy):
    """
    Follow a pre-defined circular trajectory centered above the target object.
    """

    def __init__(self):
        super().__init__()
        self.r = 0.06
        self.h = 0.3
        self.duration = 6.0
        self.m = scipy.interpolate.interp1d([0, self.duration], [np.pi, 3.0 * np.pi])

    def activate(self, bbox):
        super().activate(bbox)
        self.tic = rospy.Time.now()
        self.circle_center = (bbox.min + bbox.max) / 2.0
        self.circle_center[2] += self.h

    def update(self):
        elapsed_time = (rospy.Time.now() - self.tic).to_sec()
        if elapsed_time > self.duration:
            self.best_grasp = self._predict_best_grasp()
            self.done = True
        else:
            self._integrate_latest_image()
            t = self.m(elapsed_time)
            eye = self.circle_center + np.r_[self.r * np.cos(t), self.r * np.sin(t), 0]
            center = (self.bbox.min + self.bbox.max) / 2.0
            up = np.r_[1.0, 0.0, 0.0]
            target = self.T_B_task * (self.T_EE_cam * look_at(eye, center, up)).inv()
            return target


class AlignmentView(BasePolicy):
    """
    Align the camera with an initial grasp prediction as proposed in (Gualtieri, 2017).
    """

    def activate(self, bbox):
        super().activate(bbox)
        self._integrate_latest_image()
        self.best_grasp = self._predict_best_grasp()
        if self.best_grasp:
            R, t = self.best_grasp.rotation, self.best_grasp.translation
            center = t
            eye = R.apply([0.0, 0.0, -0.16]) + t
            up = np.r_[1.0, 0.0, 0.0]
            self.target = (self.T_EE_cam * look_at(eye, center, up)).inv()
        else:
            self.done = True

    def update(self):
        current = tf.lookup(self.base_frame, self.ee_frame)
        error = current.translation - self.target.translation

        if np.linalg.norm(error) < 0.01:
            self.best_grasp = self._predict_best_grasp()
            self.done = True
        else:
            self._integrate_latest_image()
            return self.target
