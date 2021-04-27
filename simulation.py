import pybullet as p

from robot_tools.btsim import *
from robot_tools.spatial import Rotation, Transform


class BtPandaEnv(BtBaseEnv):
    def __init__(self, gui=True, sleep=True):
        super().__init__(gui, sleep)
        self.arm = BtPandaArm()
        self.gripper = BtPandaGripper(self.arm)
        self.camera = BtCamera(self.arm, 9, 320, 240, 1.047, 0.1, 1.0)
        self.T_W_B = Transform(Rotation.identity(), np.r_[-0.6, 0.0, 0.4])
        self.load_table()
        self.load_robot()
        self.load_objects()

    def reset(self):
        q = self.arm.configurations["ready"]
        for i, q_i in enumerate(q):
            p.resetJointState(self.arm.uid, i, q_i, 0)

    def load_table(self):
        p.loadURDF("plane.urdf")
        p.loadURDF(
            "table/table.urdf",
            baseOrientation=Rotation.from_rotvec(np.array([0, 0, np.pi / 2])).as_quat(),
            useFixedBase=True,
        )

    def load_robot(self):
        self.arm.load(self.T_W_B)

    def load_objects(self):
        p.loadURDF("cube_small.urdf", [-0.2, 0.0, 0.8])
