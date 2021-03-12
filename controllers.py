import numpy as np


class CartesianPoseController:
    def __init__(self, model, x0):
        self.model = model
        self.x_d = x0
        self.kp = np.ones(6) * 5.0

    def set_target(self, x_d):
        self.x_d = x_d

    def update(self, q, dq):
        x = self.model.pose(q)
        x_d = self.x_d

        err = np.zeros(6)
        err[:3] = x_d.translation - x.translation
        err[3:] = (x_d.rotation * x.rotation.inv()).as_rotvec()

        J = self.model.jacobian(q)
        J_pinv = np.linalg.pinv(J)
        cmd = np.dot(J_pinv, self.kp * err)

        return cmd
