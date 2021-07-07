import numpy as np

from active_grasp.policy import BasePolicy
from robot_utils.ros import tf
from vgn.utils import look_at


class SingleViewBaseline(BasePolicy):
    """
    Process a single image from the initial viewpoint.
    """

    def update(self):
        self.integrate_latest_image()
        self.draw_scene_cloud()
        self.best_grasp = self.predict_best_grasp()
        self.done = True


class TopBaseline(BasePolicy):
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
            self.best_grasp = self.predict_best_grasp()
            self.done = True
        else:
            self.integrate_latest_image()
            self.draw_scene_cloud()
            self.draw_camera_path()
            return self.target
