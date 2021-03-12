import numpy as np


class CartesianPoseController:
    def __init__(self, model):
        self.model = model
        # self.x_d = x0

    def set_target(self, pose):
        self.x_d = pose.translation

    def update(self, q, dq):
        t = self.model.pose(q).translation
        J = self.model.jacobian(q)[:3, :]
        J_pinv = np.linalg.pinv(J)

        err = 2.0 * (self.x_d - t)
        cmd = np.dot(J_pinv, err)

        return cmd
