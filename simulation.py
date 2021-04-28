import pybullet as p

from robot_utils.btsim import *
from robot_utils.spatial import Rotation, Transform


class BtPandaEnv(BtBaseEnv):
    def __init__(self, gui=True, sleep=True):
        super().__init__(gui, sleep)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(1.4, 50, -35, [0.0, 0.0, 0.6])
        self.arm = BtPandaArm()
        self.gripper = BtPandaGripper(self.arm)
        self.camera = BtCamera(
            self.arm, 11, 320, 240, 1.047, 0.1, 1.0
        )  # link 11 corresponds to cam_optical_frame
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
