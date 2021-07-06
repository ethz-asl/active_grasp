from pathlib import Path
import pybullet as p
import rospkg

from robot_utils.bullet import *
from robot_utils.controllers import CartesianPoseController
from robot_utils.spatial import Rotation, Transform


class Simulation(BtSim):
    def __init__(self, gui=True):
        super().__init__(gui=gui, sleep=False)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(1.4, 50, -35, [0.0, 0.0, 0.6])

        self.object_uids = []

        self.find_object_urdfs()
        self.load_table()
        self.load_robot()
        self.load_controller()

        self.reset()

    def find_object_urdfs(self):
        rospack = rospkg.RosPack()
        root = Path(rospack.get_path("vgn")) / "assets/urdfs/packed/test"
        self.urdfs = [str(f) for f in root.iterdir() if f.suffix == ".urdf"]

    def load_table(self):
        p.loadURDF("plane.urdf")
        ori = Rotation.from_rotvec(np.array([0, 0, np.pi / 2])).as_quat()
        p.loadURDF("table/table.urdf", baseOrientation=ori, useFixedBase=True)
        self.length = 0.3
        self.origin = [-0.3, -0.5 * self.length, 0.5]

    def load_robot(self):
        self.T_W_B = Transform(Rotation.identity(), np.r_[-0.6, 0.0, 0.4])
        self.arm = BtPandaArm(self.T_W_B)
        self.gripper = BtPandaGripper(self.arm)
        self.model = Model(self.arm.urdf_path, self.arm.base_frame, self.arm.ee_frame)
        self.camera = BtCamera(320, 240, 1.047, 0.1, 1.0, self.arm.uid, 11)

    def load_controller(self):
        self.controller = CartesianPoseController(self.model, self.arm.ee_frame, None)

    def reset(self):
        self.remove_all_objects()
        self.set_initial_arm_configuration()
        urdfs = np.random.choice(self.urdfs, 4)
        origin = np.r_[self.origin[:2], 0.625]
        self.random_object_arrangement(urdfs, origin, self.length, 0.8)

    def set_initial_arm_configuration(self):
        q = self.arm.configurations["ready"]
        q[0] = np.deg2rad(np.random.uniform(-10, 10))
        q[5] = np.deg2rad(np.random.uniform(90, 105))
        for i, q_i in enumerate(q):
            p.resetJointState(self.arm.uid, i, q_i, 0)
        p.resetJointState(self.arm.uid, 9, 0.04, 0)
        p.resetJointState(self.arm.uid, 10, 0.04, 0)
        x0 = self.model.pose(self.arm.ee_frame, q)
        self.controller.x_d = x0

    def load_object(self, urdf, ori, pos, scale=1.0):
        uid = p.loadURDF(str(urdf), pos, ori.as_quat(), globalScaling=scale)
        self.object_uids.append(uid)
        return uid

    def remove_object(self, uid):
        p.removeBody(uid)
        self.object_uids.remove(uid)

    def remove_all_objects(self):
        for uid in list(self.object_uids):
            self.remove_object(uid)

    def random_object_arrangement(self, urdfs, origin, length, scale=1.0, attempts=10):
        for urdf in urdfs:
            # Load the object
            uid = self.load_object(urdf, Rotation.identity(), [0.0, 0.0, 0.0], scale)
            lower, upper = p.getAABB(uid)
            z_offset = 0.5 * (upper[2] - lower[2]) + 0.002
            state_id = p.saveState()
            for _ in range(attempts):
                # Try to place the object without collision
                ori = Rotation.from_rotvec([0.0, 0.0, np.random.uniform(0, 2 * np.pi)])
                offset = np.r_[np.random.uniform(0.2, 0.8, 2) * length, z_offset]
                p.resetBasePositionAndOrientation(uid, origin + offset, ori.as_quat())
                self.step()
                if not p.getContactPoints(uid):
                    break
                else:
                    p.restoreState(stateId=state_id)
            else:
                # No placement found, remove the object
                self.remove_object(uid)


class CartesianPoseController:
    def __init__(self, model, frame, x0):
        self._model = model
        self._frame = frame

        self.kp = np.ones(6) * 4.0
        self.max_linear_vel = 0.2
        self.max_angular_vel = 1.57

        self.x_d = x0

    def update(self, q):
        x = self._model.pose(self._frame, q)
        error = np.zeros(6)
        error[:3] = self.x_d.translation - x.translation
        error[3:] = (self.x_d.rotation * x.rotation.inv()).as_rotvec()
        dx = self._limit_rate(self.kp * error)
        J_pinv = np.linalg.pinv(self._model.jacobian(self._frame, q))
        cmd = np.dot(J_pinv, dx)
        return cmd

    def _limit_rate(self, dx):
        linear, angular = dx[:3], dx[3:]
        linear = np.clip(linear, -self.max_linear_vel, self.max_linear_vel)
        angular = np.clip(angular, -self.max_angular_vel, self.max_angular_vel)
        return np.r_[linear, angular]
