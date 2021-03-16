import numpy as np


class CartesianPoseController:
    def __init__(self, robot, model, rate):
        self.robot = robot
        self.model = model
        self.rate = rate
        self.x_d = None
        self.kp = np.ones(6) * 5.0

    def set_target(self, pose):
        self.x_d = pose

    def update(self):
        q, _ = self.robot.get_state()

        x = self.model.pose(q)
        x_d = self.x_d

        err = np.zeros(6)
        err[:3] = x_d.translation - x.translation
        err[3:] = (x_d.rotation * x.rotation.inv()).as_rotvec()

        J = self.model.jacobian(q)
        J_pinv = np.linalg.pinv(J)
        cmd = np.dot(J_pinv, self.kp * err)

        self.robot.set_desired_joint_velocities(cmd)
