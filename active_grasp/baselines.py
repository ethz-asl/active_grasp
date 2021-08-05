import numpy as np
import scipy.interpolate
import rospy

from .policy import BasePolicy
from vgn.utils import look_at


class SingleView(BasePolicy):
    """
    Process a single image from the initial viewpoint.
    """

    def update(self, img, extrinsic):
        self.integrate_img(img, extrinsic)
        self.best_grasp = self.predict_best_grasp()
        self.done = True


class TopView(BasePolicy):
    """
    Move the camera to a top-down view of the target object.
    """

    def activate(self, bbox):
        super().activate(bbox)
        eye = np.r_[self.center[:2], self.center[2] + 0.3]
        up = np.r_[1.0, 0.0, 0.0]
        self.target = look_at(eye, self.center, up)

    def update(self, img, extrinsic):
        self.integrate_img(img, extrinsic)
        error = extrinsic.translation - self.target.translation
        if np.linalg.norm(error) < 0.01:
            self.best_grasp = self.predict_best_grasp()
            self.done = True
        return self.target


class RandomView(BasePolicy):
    """
    Move the camera to a random viewpoint on a circle centered above the target.
    """

    def __init__(self, intrinsic):
        super().__init__(intrinsic)
        self.r = 0.06  # radius of the circle
        self.h = 0.3  # distance above bbox center

    def activate(self, bbox):
        super().activate(bbox)
        t = np.random.uniform(np.pi, 3.0 * np.pi)
        eye = self.center + np.r_[self.r * np.cos(t), self.r * np.sin(t), self.h]
        up = np.r_[1.0, 0.0, 0.0]
        self.target = look_at(eye, self.center, up)

    def update(self, img, extrinsic):
        self.integrate_img(img, extrinsic)
        error = extrinsic.translation - self.target.translation
        if np.linalg.norm(error) < 0.01:
            self.best_grasp = self.predict_best_grasp()
            self.done = True
        return self.target


class FixedTrajectory(BasePolicy):
    """
    Follow a pre-defined circular trajectory centered above the target object.
    """

    def __init__(self, intrinsic):
        super().__init__(intrinsic)
        self.r = 0.08
        self.h = 0.3
        self.duration = 6.0
        self.m = scipy.interpolate.interp1d([0, self.duration], [np.pi, 3.0 * np.pi])

    def activate(self, bbox):
        super().activate(bbox)
        self.tic = rospy.Time.now()

    def update(self, img, extrinsic):
        self.integrate_img(img, extrinsic)
        elapsed_time = (rospy.Time.now() - self.tic).to_sec()
        if elapsed_time > self.duration:
            self.best_grasp = self.predict_best_grasp()
            self.done = True
        else:
            t = self.m(elapsed_time)
            eye = self.center + np.r_[self.r * np.cos(t), self.r * np.sin(t), self.h]
            up = np.r_[1.0, 0.0, 0.0]
            target = look_at(eye, self.center, up)
            return target


class AlignmentView(BasePolicy):
    """
    Align the camera with an initial grasp prediction as proposed in (Gualtieri, 2017).
    """

    def activate(self, bbox):
        super().activate(bbox)
        self.target = None

    def update(self, img, extrinsic):
        self.integrate_img(img, extrinsic)

        if not self.target:
            grasp = self.predict_best_grasp()
            if not grasp:
                self.done = True
                return
            R, t = grasp.pose.rotation, grasp.pose.translation
            eye = R.apply([0.0, 0.0, -0.16]) + t
            center = t
            up = np.r_[1.0, 0.0, 0.0]
            self.target = look_at(eye, center, up)

        error = extrinsic.translation - self.target.translation
        if np.linalg.norm(error) < 0.01:
            self.best_grasp = self.predict_best_grasp()
            self.done = True
        return self.target
