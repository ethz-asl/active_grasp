from pathlib import Path
import pybullet as p
import pybullet_data
import rospkg

from active_grasp.bbox import AABBox
from robot_helpers.bullet import *
from robot_helpers.io import load_yaml
from robot_helpers.model import KDLModel
from robot_helpers.spatial import Rotation
from vgn.perception import UniformTSDFVolume
from vgn.utils import find_urdfs, view_on_sphere
from vgn.detection import VGN, select_local_maxima

# import vgn.visualizer as vis

rospack = rospkg.RosPack()
pkg_root = Path(rospack.get_path("active_grasp"))
urdfs_dir = pkg_root / "assets"


class Simulation:
    """Robot is placed s.t. world and base frames are the same"""

    def __init__(self, gui, scene_id, vgn_path):
        self.configure_physics_engine(gui, 60, 4)
        self.configure_visualizer()
        self.seed()
        self.load_robot()
        self.load_vgn(Path(vgn_path))
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

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed) if seed else np.random

    def load_robot(self):
        panda_urdf_path = urdfs_dir / "franka/panda_arm_hand.urdf"
        self.arm = BtPandaArm(panda_urdf_path)
        self.gripper = BtPandaGripper(self.arm)
        self.model = KDLModel.from_urdf_file(
            panda_urdf_path, self.arm.base_frame, self.arm.ee_frame
        )
        self.camera = BtCamera(320, 240, 0.96, 0.01, 1.0, self.arm.uid, 11)

    def load_vgn(self, model_path):
        self.vgn = VGN(model_path)

    def reset(self):
        valid = False
        while not valid:
            self.set_arm_configuration([0.0, -1.39, 0.0, -2.36, 0.0, 1.57, 0.79])
            self.scene.clear()
            q = self.scene.generate(self.rng)
            self.set_arm_configuration(q)
            uid = self.select_target()
            bbox = self.get_target_bbox(uid)
            valid = self.check_for_grasps(bbox)
        return bbox

    def set_arm_configuration(self, q):
        for i, q_i in enumerate(q):
            p.resetJointState(self.arm.uid, i, q_i, 0)
        p.resetJointState(self.arm.uid, 9, 0.04, 0)
        p.resetJointState(self.arm.uid, 10, 0.04, 0)
        self.gripper.set_desired_width(0.4)

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

    def check_for_grasps(self, bbox):
        origin = Transform.from_translation(self.scene.origin)
        origin.translation[2] -= 0.05
        center = Transform.from_translation(self.scene.center)

        # First, reconstruct the scene from many views
        tsdf = UniformTSDFVolume(self.scene.length, 40)
        r = 2.0 * self.scene.length
        theta = np.pi / 4.0
        phis = np.linspace(0.0, 2.0 * np.pi, 5)
        for view in [view_on_sphere(center, r, theta, phi) for phi in phis]:
            depth_img = self.camera.get_image(view)[1]
            tsdf.integrate(depth_img, self.camera.intrinsic, view.inv() * origin)
        voxel_size, tsdf_grid = tsdf.voxel_size, tsdf.get_grid()

        # Then check whether VGN can find any grasps on the target
        out = self.vgn.predict(tsdf_grid)
        grasps, qualities = select_local_maxima(voxel_size, out, threshold=0.9)

        # vis.scene_cloud(voxel_size, tsdf.get_scene_cloud())
        # vis.grasps(grasps, qualities, 0.05)
        # vis.show()

        for grasp in grasps:
            pose = origin * grasp.pose
            tip = pose.rotation.apply([0, 0, 0.05]) + pose.translation
            if bbox.is_inside(tip):
                return True
        return False

    def step(self):
        p.stepSimulation()


class Scene:
    def __init__(self):
        self.support_urdf = urdfs_dir / "plane/model.urdf"
        self.support_uid = -1
        self.object_uids = []

    def clear(self):
        self.remove_support()
        self.remove_all_objects()

    def generate(self, rng):
        raise NotImplementedError

    def add_support(self, pos):
        self.support_uid = p.loadURDF(str(self.support_urdf), pos, globalScaling=0.3)

    def remove_support(self):
        p.removeBody(self.support_uid)

    def add_object(self, urdf, ori, pos, scale=1.0):
        uid = p.loadURDF(str(urdf), pos, ori.as_quat(), globalScaling=scale)
        self.object_uids.append(uid)
        return uid

    def remove_object(self, uid):
        p.removeBody(uid)
        self.object_uids.remove(uid)

    def remove_all_objects(self):
        for uid in list(self.object_uids):
            self.remove_object(uid)


class YamlScene(Scene):
    def __init__(self, config_name):
        super().__init__()
        self.config_path = pkg_root / "cfg/sim" / config_name

    def load_config(self):
        self.scene = load_yaml(self.config_path)
        self.center = np.asarray(self.scene["center"])
        self.length = 0.3
        self.origin = self.center - np.r_[0.5 * self.length, 0.5 * self.length, 0.0]

    def generate(self, rng):
        self.load_config()
        self.add_support(self.center)
        for object in self.scene["objects"]:
            urdf = urdfs_dir / object["object_id"] / "model.urdf"
            ori = Rotation.from_euler("xyz", object["rpy"], degrees=True)
            pos = self.center + np.asarray(object["xyz"])
            scale = object.get("scale", 1)
            if randomize := object.get("randomize", False):
                angle = rng.uniform(-randomize["rot"], randomize["rot"])
                ori = Rotation.from_euler("z", angle, degrees=True) * ori
                b = np.asarray(randomize["pos"])
                pos += rng.uniform(-b, b)
            self.add_object(urdf, ori, pos, scale)
        for _ in range(60):
            p.stepSimulation()
        return self.scene["q"]


class RandomScene(Scene):
    def __init__(self):
        super().__init__()
        self.center = np.r_[0.5, 0.0, 0.2]
        self.length = 0.3
        self.origin = self.center - np.r_[0.5 * self.length, 0.5 * self.length, 0.0]
        self.object_urdfs = find_urdfs(urdfs_dir / "test")

    def generate(self, rng, object_count=4, attempts=10):
        self.add_support(self.center)
        urdfs = rng.choice(self.object_urdfs, object_count)
        for urdf in urdfs:
            scale = rng.uniform(0.8, 1.0)
            uid = self.add_object(urdf, Rotation.identity(), np.zeros(3), scale)
            lower, upper = p.getAABB(uid)
            z_offset = 0.5 * (upper[2] - lower[2]) + 0.002
            state_id = p.saveState()
            for _ in range(attempts):
                # Try to place and check for collisions
                ori = Rotation.from_euler("z", rng.uniform(0, 2 * np.pi))
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
        q = [0.0, -1.39, 0.0, -2.36, 0.0, 1.57, 0.79]
        q += rng.uniform(-0.08, 0.08, 7)
        return q


def get_scene(scene_id):
    if scene_id.endswith(".yaml"):
        return YamlScene(scene_id)
    elif scene_id == "random":
        return RandomScene()

    else:
        raise ValueError("Unknown scene {}.".format(scene_id))
