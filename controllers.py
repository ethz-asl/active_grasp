import numpy as np


class CartesianPoseController:
    def __init__(self, model):
        self.model = model

    def set_target(self, pose):
        self.target = pose.translation

    def update(self, q, dq):
        t = self.model.pose().translation
        J = self.model.jacobian(q)
        J_pinv = np.linalg.pinv(J)

        err = 2.0 * (self.target - t)
        cmd = np.dot(J_pinv, err)

        return cmd
