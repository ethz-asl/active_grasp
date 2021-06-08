from pathlib import Path
import pybullet as p
import rospkg

from robot_utils.bullet import *
from robot_utils.spatial import Rotation, Transform


class Simulation(BtManipulationSim):
    def __init__(self, gui=True):
        super().__init__(gui=gui, sleep=False)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(1.4, 50, -35, [0.0, 0.0, 0.6])

        self.find_object_urdfs()
        self.add_table()
        self.add_robot()

    def find_object_urdfs(self):
        rospack = rospkg.RosPack()
        root = Path(rospack.get_path("vgn")) / "data/urdfs/packed/test"
        self.urdfs = [str(f) for f in root.iterdir() if f.suffix == ".urdf"]

    def add_table(self):
        p.loadURDF("plane.urdf")
        ori = Rotation.from_rotvec(np.array([0, 0, np.pi / 2])).as_quat()
        p.loadURDF("table/table.urdf", baseOrientation=ori, useFixedBase=True)
        self.length = 0.3
        self.origin = [-0.3, -0.5 * self.length, 0.5]

    def add_robot(self):
        self.T_W_B = Transform(Rotation.identity(), np.r_[-0.6, 0.0, 0.5])
        self.arm = BtPandaArm(self.T_W_B)
        self.gripper = BtPandaGripper(self.arm)
        self.camera = BtCamera(320, 240, 1.047, 0.1, 1.0, self.arm.uid, 11)

    def reset(self):
        self.remove_all_objects()
        self.set_initial_arm_configuration()
        urdfs = np.random.choice(self.urdfs, 4)
        origin = np.r_[self.origin[:2], 0.625]
        self.add_random_arrangement(urdfs, origin, self.length, 0.8)

    def set_initial_arm_configuration(self):
        q = self.arm.configurations["ready"]
        q[0] = np.deg2rad(np.random.uniform(-10, 10))
        q[5] = np.deg2rad(np.random.uniform(90, 105))
        for i, q_i in enumerate(q):
            p.resetJointState(self.arm.uid, i, q_i, 0)
