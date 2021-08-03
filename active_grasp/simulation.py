from pathlib import Path
import pybullet as p
import pybullet_data
import rospkg

from active_grasp.bbox import AABBox
from robot_helpers.bullet import *
from robot_helpers.model import Model
from robot_helpers.spatial import Rotation, Transform


class Simulation:
    def __init__(self, gui):
        self.configure_physics_engine(gui, 60, 4)
        self.configure_visualizer()
        self.find_urdfs()
        self.load_table()
        self.load_robot()
        self.load_controller()
        self.object_uids = []

    def configure_physics_engine(self, gui, rate, sub_step_count):
        self.rate = rate
        self.dt = 1.0 / self.rate
        p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(fixedTimeStep=self.dt, numSubSteps=sub_step_count)
        p.setGravity(0.0, 0.0, -9.81)

    def configure_visualizer(self):
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(1.4, 50, -35, [0.0, 0.0, 0.6])

    def find_urdfs(self):
        rospack = rospkg.RosPack()
        assets_path = Path(rospack.get_path("active_grasp")) / "assets"
        self.panda_urdf = assets_path / "urdfs/franka/panda_arm_hand.urdf"
        root = Path(rospack.get_path("vgn")) / "assets/urdfs/packed/test"
        self.urdfs = [str(f) for f in root.iterdir() if f.suffix == ".urdf"]

    def load_table(self):
        p.loadURDF("plane.urdf")
        ori = Rotation.from_rotvec(np.array([0, 0, np.pi / 2])).as_quat()
        p.loadURDF("table/table.urdf", baseOrientation=ori, useFixedBase=True)
        self.length = 0.3
        self.origin = [-0.3, -0.5 * self.length, 0.5]

    def load_robot(self):
        self.T_world_base = Transform.translation(np.r_[-0.6, 0.0, 0.4])
        self.arm = BtPandaArm(self.panda_urdf, self.T_world_base)
        self.gripper = BtPandaGripper(self.arm)
        self.model = Model(self.panda_urdf, self.arm.base_frame, self.arm.ee_frame)
        self.camera = BtCamera(320, 240, 1.047, 0.1, 1.0, self.arm.uid, 11)

    def load_controller(self):
        q, _ = self.arm.get_state()
        x0 = self.model.pose(self.arm.ee_frame, q)
        self.controller = CartesianPoseController(self.model, self.arm.ee_frame, x0)

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.remove_all_objects()
        self.set_initial_arm_configuration()
        self.load_random_object_arrangement()
        uid = self.select_target()
        return self.get_target_bbox(uid)

    def step(self):
        p.stepSimulation()

    def set_initial_arm_configuration(self):
        q = [
            self.rng.uniform(-0.17, 0.17),  # 0.0
            self.rng.uniform(-0.96, -0.62),  # -0.79,
            self.rng.uniform(-0.17, 0.17),  # 0.0
            self.rng.uniform(-2.36, -2.19),  # -2.36,
            0.0,
            self.rng.uniform(1.57, 1.91),  # 1.57
            self.rng.uniform(0.62, 0.96),  # 0.79,
        ]
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

    def load_random_object_arrangement(self, attempts=10):
        origin = np.r_[self.origin[:2], 0.625]
        scale = 0.8
        urdfs = self.rng.choice(self.urdfs, 4)
        for urdf in urdfs:
            # Load the object
            uid = self.load_object(urdf, Rotation.identity(), [0.0, 0.0, 0.0], scale)
            lower, upper = p.getAABB(uid)
            z_offset = 0.5 * (upper[2] - lower[2]) + 0.002
            state_id = p.saveState()
            for _ in range(attempts):
                # Try to place the object without collision
                ori = Rotation.from_rotvec([0.0, 0.0, self.rng.uniform(0, 2 * np.pi)])
                offset = np.r_[self.rng.uniform(0.2, 0.8, 2) * self.length, z_offset]
                p.resetBasePositionAndOrientation(uid, origin + offset, ori.as_quat())
                self.step()
                if not p.getContactPoints(uid):
                    break
                else:
                    p.restoreState(stateId=state_id)
            else:
                # No placement found, remove the object
                self.remove_object(uid)

    def select_target(self):
        _, _, mask = self.camera.get_image()
        uids, counts = np.unique(mask, return_counts=True)
        mask = np.isin(uids, self.object_uids)  # remove ids of the floor, table, etc
        uids, counts = uids[mask], counts[mask]
        target_uid = uids[np.argmin(counts)]
        p.changeVisualShape(target_uid, -1, rgbaColor=[1, 0, 0, 1])
        return target_uid

    def get_target_bbox(self, uid):
        aabb_min, aabb_max = p.getAABB(uid)
        # Transform the coordinates to base_frame
        aabb_min = np.array(aabb_min) - self.T_world_base.translation
        aabb_max = np.array(aabb_max) - self.T_world_base.translation
        return AABBox(aabb_min, aabb_max)


class CartesianPoseController:
    def __init__(self, model, frame, x0):
        self.model = model
        self.frame = frame

        self.kp = np.ones(6) * 4.0
        self.max_linear_vel = 0.1
        self.max_angular_vel = 1.57

        self.x_d = x0

    def update(self, q):
        x = self.model.pose(self.frame, q)
        error = np.zeros(6)
        error[:3] = self.x_d.translation - x.translation
        error[3:] = (self.x_d.rotation * x.rotation.inv()).as_rotvec()
        dx = self.limit_rate(self.kp * error)
        J_pinv = np.linalg.pinv(self.model.jacobian(self.frame, q))
        cmd = np.dot(J_pinv, dx)
        return cmd

    def limit_rate(self, dx):
        linear, angular = dx[:3], dx[3:]
        linear = np.clip(linear, -self.max_linear_vel, self.max_linear_vel)
        angular = np.clip(angular, -self.max_angular_vel, self.max_angular_vel)
        return np.r_[linear, angular]
