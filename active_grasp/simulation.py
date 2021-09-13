from pathlib import Path
import pybullet as p
import pybullet_data
import yaml
import rospkg

from active_grasp.bbox import AABBox
from robot_helpers.bullet import *
from robot_helpers.model import KDLModel
from robot_helpers.spatial import Rotation


rospack = rospkg.RosPack()


class Simulation:
    """Robot is placed s.t. world and base frames are the same"""

    def __init__(self, gui, scene_id):
        self.configure_physics_engine(gui, 60, 4)
        self.configure_visualizer()
        self.load_robot()
        self.scene = get_scene(scene_id)

    def configure_physics_engine(self, gui, rate, sub_step_count):
        self.rate = rate
        self.dt = 1.0 / self.rate
        p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(fixedTimeStep=self.dt, numSubSteps=sub_step_count)
        p.setGravity(0.0, 0.0, -9.81)

    def configure_visualizer(self):
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(1.2, 30, -30, [0.4, 0.0, 0.2])

    def load_robot(self):
        path = Path(rospack.get_path("active_grasp"))
        panda_urdf_path = path / "assets/urdfs/franka/panda_arm_hand.urdf"
        self.arm = BtPandaArm(panda_urdf_path)
        self.gripper = BtPandaGripper(self.arm)
        self.model = KDLModel.from_urdf_file(
            panda_urdf_path, self.arm.base_frame, self.arm.ee_frame
        )
        self.camera = BtCamera(320, 240, 0.96, 0.01, 1.0, self.arm.uid, 11)

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.set_arm_configuration([0.0, -0.79, 0.0, -2.356, 0.0, 1.57, 0.79])
        self.scene.reset(rng=self.rng)
        q = self.scene.sample_initial_configuration(self.rng)
        self.set_arm_configuration(q)
        uid = self.select_target()
        bbox = self.get_target_bbox(uid)
        return bbox

    def set_arm_configuration(self, q):
        for i, q_i in enumerate(q):
            p.resetJointState(self.arm.uid, i, q_i, 0)
        p.resetJointState(self.arm.uid, 9, 0.04, 0)
        p.resetJointState(self.arm.uid, 10, 0.04, 0)

    def select_target(self):
        _, _, mask = self.camera.get_image()
        uids, counts = np.unique(mask, return_counts=True)
        mask = np.isin(uids, self.scene.object_uids)  # remove ids of the floor, etc
        uids, counts = uids[mask], counts[mask]
        target_uid = uids[np.argmin(counts)]
        p.changeVisualShape(target_uid, -1, rgbaColor=[1, 0, 0, 1])
        return target_uid

    def get_target_bbox(self, uid):
        aabb_min, aabb_max = p.getAABB(uid)
        return AABBox(aabb_min, aabb_max)

    def step(self):
        p.stepSimulation()


class Scene:
    def __init__(self):
        self.vgn_urdfs_dir = Path(rospack.get_path("vgn")) / "assets/urdfs"
        self.ycb_urdfs_dir = Path(rospack.get_path("urdf_zoo")) / "models/ycb"
        self.support_urdf = self.vgn_urdfs_dir / "setup/plane.urdf"
        self.support_uid = -1
        self.object_uids = []

    def load_support(self, pos):
        self.support_uid = p.loadURDF(str(self.support_urdf), pos, globalScaling=0.3)

    def remove_support(self):
        p.removeBody(self.support_uid)

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

    def reset(self, rng):
        self.remove_support()
        self.remove_all_objects()
        self.load(rng)

    def load(self, rng):
        raise NotImplementedError

    def get_ycb_urdf_path(self, model_name):
        return self.ycb_urdfs_dir / model_name / "model.urdf"


def find_urdfs(root):
    # Scans a dir for URDF assets
    return [str(f) for f in root.iterdir() if f.suffix == ".urdf"]


class RandomScene(Scene):
    def __init__(self):
        super().__init__()
        self.center = np.r_[0.5, 0.0, 0.1]
        self.length = 0.3
        self.origin = self.center - np.r_[0.5 * self.length, 0.5 * self.length, 0.0]
        self.object_urdfs = find_urdfs(self.vgn_urdfs_dir / "packed" / "test")

    def load(self, rng, attempts=10):
        self.load_support(self.center)
        urdfs, scale = rng.choice(self.object_urdfs, 4), 0.8
        for urdf in urdfs:
            uid = self.load_object(urdf, Rotation.identity(), np.zeros(3), scale)
            lower, upper = p.getAABB(uid)
            z_offset = 0.5 * (upper[2] - lower[2]) + 0.002
            state_id = p.saveState()
            for _ in range(attempts):
                # Try to place and check for collisions
                ori = Rotation.from_rotvec([0.0, 0.0, rng.uniform(0, 2 * np.pi)])
                pos = np.r_[rng.uniform(0.2, 0.8, 2) * self.length, z_offset]
                p.resetBasePositionAndOrientation(uid, self.origin + pos, ori.as_quat())
                p.stepSimulation()
                if not p.getContactPoints(uid):
                    break
                else:
                    p.restoreState(stateId=state_id)
            else:
                # No placement found, remove the object
                self.remove_object(uid)

    def sample_initial_configuration(self, rng):
        # q = [0.0, -0.79, 0.0, -2.36, 0.0, 1.57, 0.79]
        q = [0.0, -0.96, 0.0, -2.09, 0.0, 1.66, 0.79]
        q += rng.uniform(-0.08, 0.08, 7)
        return q


class CustomScene(Scene):
    def __init__(self, config_name):
        super().__init__()
        self.config_path = (
            Path(rospack.get_path("active_grasp")) / "cfg" / "scenes" / config_name
        )

    def load_config(self):
        with self.config_path.open("r") as f:
            self.scene = yaml.load(f)
        self.center = np.asarray(self.scene["center"])
        self.length = 0.3
        self.origin = self.center - np.r_[0.5 * self.length, 0.5 * self.length, 0.0]

    def load(self, rng):
        self.load_config()
        self.load_support(self.center)
        for object in self.scene["objects"]:
            self.load_object(
                self.get_ycb_urdf_path(object["object_id"]),
                Rotation.from_euler("xyz", object["rpy"], degrees=True),
                self.center + np.asarray(object["xyz"]),
                object.get("scale", 1),
            )
        for _ in range(60):
            p.stepSimulation()

    def sample_initial_configuration(self, rng):
        return self.scene["q"]


def get_scene(scene_id):
    if scene_id == "random":
        return RandomScene()
    elif scene_id.endswith(".yaml"):
        return CustomScene(scene_id)
    else:
        raise ValueError("Unknown scene {}.".format(scene_id))
