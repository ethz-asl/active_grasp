from pathlib import Path
import pybullet as p
import rospkg

from robot_tools.bullet import *
from robot_tools.spatial import Rotation, Transform
from robot_tools.utils import scan_dir_for_urdfs


class Simulation(BtManipulationSim):
    def __init__(self, gui=True, sleep=True):
        super().__init__(gui, sleep)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(1.4, 50, -35, [0.0, 0.0, 0.6])

        self.find_object_urdfs()
        self.add_table()
        self.add_robot()

    def find_object_urdfs(self):
        rospack = rospkg.RosPack()
        root = Path(rospack.get_path("vgn")) / "data/urdfs/packed/test"
        self.urdfs = scan_dir_for_urdfs(root)

    def add_table(self):
        p.loadURDF("plane.urdf")
        ori = Rotation.from_rotvec(np.array([0, 0, np.pi / 2])).as_quat()
        p.loadURDF("table/table.urdf", baseOrientation=ori, useFixedBase=True)
        self.length = 0.3
        self.origin = [-0.2, -0.5 * self.length, 0.5]

    def add_robot(self):
        self.T_W_B = Transform(Rotation.identity(), np.r_[-0.6, 0.0, 0.5])
        self.arm = BtPandaArm(self.T_W_B)
        self.gripper = BtPandaGripper(self.arm)
        self.camera = BtCamera(320, 240, 1.047, 0.1, 1.0, self.arm.uid, 11)

    def reset(self):
        self.remove_all_objects()
        urdfs = np.random.choice(self.urdfs, 4)
        self.add_random_arrangement(urdfs, np.r_[self.origin[:2], 0.625], self.length)
        self.set_initial_arm_configuration()

    def set_initial_arm_configuration(self):
        q = self.arm.configurations["ready"]
        q[0] = np.deg2rad(np.random.uniform(-10, 10))
        q[5] = np.deg2rad(np.random.uniform(90, 105))
        for i, q_i in enumerate(q):
            p.resetJointState(self.arm.uid, i, q_i, 0)
