import cv_bridge
import numpy as np
from pathlib import Path
import rospy
import scipy.interpolate

from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, CameraInfo
import std_srvs.srv

from robot_tools.spatial import Rotation, Transform
from robot_tools.ros.conversions import *
from robot_tools.ros.control import ControllerManagerClient
from robot_tools.ros.panda import PandaGripperClient
from robot_tools.ros.tf import TF2Client
from robot_tools.perception import *
from vgn import vis
from vgn.detection import VGN, compute_grasps


def get_controller(name):
    if name == "single-view":
        return SingleViewBaseline()
    elif name == "fixed-trajectory":
        return FixedTrajectoryBaseline()
    elif name == "mvp":
        return MultiViewPicking()
    else:
        raise ValueError("{} policy does not exist.".format(name))


class BaseController:
    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id")
        self.length = rospy.get_param("~length")

        self.cv_bridge = cv_bridge.CvBridge()
        self.tf = TF2Client()

        self.reset_client = rospy.ServiceProxy("/reset", std_srvs.srv.Trigger)

        self.tsdf = UniformTSDFVolume(0.3, 40)
        self.vgn = VGN(Path(rospy.get_param("vgn/model")))

        self.setup_robot_connection()
        self.setup_camera_connection()
        self.lookup_transforms()

    def setup_robot_connection(self):
        self.base_frame_id = rospy.get_param("~base_frame_id")
        self.ee_frame_id = rospy.get_param("~ee_frame_id")
        self.ee_grasp_offset = Transform.from_list(rospy.get_param("~ee_grasp_offset"))
        self.target_pose_pub = rospy.Publisher("/command", Pose, queue_size=10)
        self.gripper = PandaGripperClient()

    def send_pose_command(self, target):
        self.target_pose_pub.publish(to_pose_msg(target))

    def setup_camera_connection(self):
        self.cam_frame_id = rospy.get_param("~camera/frame_id")
        info_topic = rospy.get_param("~camera/info_topic")
        msg = rospy.wait_for_message(info_topic, CameraInfo, rospy.Duration(2.0))
        self.intrinsic = from_camera_info_msg(msg)
        depth_topic = rospy.get_param("~camera/depth_topic")
        rospy.Subscriber(depth_topic, Image, self.sensor_cb, queue_size=1)

    def sensor_cb(self, msg):
        self.last_depth_img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32)
        self.last_extrinsic = self.tf.lookup(
            self.cam_frame_id,
            self.frame_id,
            msg.header.stamp,
            rospy.Duration(0.1),
        )

    def lookup_transforms(self):
        self.T_B_O = self.tf.lookup(self.base_frame_id, self.frame_id, rospy.Time.now())

    def run(self):
        self.reset()
        self.explore()
        self.execute_grasp()

    def reset(self):
        vis.clear()
        req = std_srvs.srv.TriggerRequest()
        self.reset_client(req)
        rospy.sleep(1.0)  # wait for states to be updated
        self.done = False

    def explore(self):
        r = rospy.Rate(self.rate)
        while not self.done:
            self.update()
            r.sleep()

    def update(self):
        raise NotImplementedError

    def execute_grasp(self):
        if not self.best_grasp:
            return
        grasp = self.best_grasp

        # Ensure that the camera is pointing forward.
        rot = grasp.pose.rotation
        if rot.as_matrix()[:, 0][0] < 0:
            grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)
        target = self.T_B_O * grasp.pose * self.ee_grasp_offset.inv()

        self.gripper.move(0.08)
        self.send_pose_command(target)
        rospy.sleep(3.0)
        self.gripper.move(0.0)
        target.translation[2] += 0.3
        self.send_pose_command(target)
        rospy.sleep(2.0)

    def predict_best_grasp(self):
        tsdf_grid = self.tsdf.get_grid()
        out = self.vgn.predict(tsdf_grid)
        score_fn = lambda g: g.pose.translation[2]
        grasps = compute_grasps(self.tsdf.voxel_size, out, score_fn)
        vis.draw_grasps(grasps, 0.05)
        return grasps[0] if len(grasps) > 0 else None


class SingleViewBaseline(BaseController):
    def __init__(self):
        super().__init__()
        self.rate = 1

    def reset(self):
        super().reset()

    def update(self):
        self.tsdf.integrate(
            self.last_depth_img,
            self.intrinsic,
            self.last_extrinsic,
        )
        cloud = self.tsdf.get_scene_cloud()
        vis.draw_points(np.asarray(cloud.points))
        self.best_grasp = self.predict_best_grasp()
        self.done = True


class FixedTrajectoryBaseline(BaseController):
    def __init__(self):
        super().__init__()
        self.rate = 10
        self.duration = 4.0
        self.radius = 0.1
        self.m = scipy.interpolate.interp1d([0, self.duration], [np.pi, 3.0 * np.pi])

    def reset(self):
        super().reset()
        self.tic = rospy.Time.now()
        timeout = rospy.Duration(0.1)
        x0 = self.tf.lookup(self.base_frame_id, self.ee_frame_id, self.tic, timeout)
        self.center = np.r_[x0.translation[0] + self.radius, x0.translation[1:]]
        self.target = x0

    def update(self):
        elapsed_time = (rospy.Time.now() - self.tic).to_sec()
        if elapsed_time > self.duration:
            self.best_grasp = self.predict_best_grasp()
            self.done = True
        else:
            self.tsdf.integrate(
                self.last_depth_img,
                self.intrinsic,
                self.last_extrinsic,
            )
            cloud = self.tsdf.get_scene_cloud()
            vis.draw_points(np.asarray(cloud.points))
            t = self.m(elapsed_time)
            self.target.translation = (
                self.center
                + np.r_[self.radius * np.cos(t), self.radius * np.sin(t), 0.0]
            )
            self.send_pose_command(self.target)


class Map:
    def __init__(self):
        pass

    def update(self):
        pass


class MultiViewPicking(BaseController):
    def __init__(self):
        super().__init__()
        self.rate = 5
        self.grid = np.zeros((40, 40, 40))

    def reset(self):
        super().reset()
        self.tic = rospy.Time.now()
        timeout = rospy.Duration(0.1)
        x0 = self.tf.lookup(self.base_frame_id, self.ee_frame_id, self.tic, timeout)
        self.center = np.r_[x0.translation[0] + self.radius, x0.translation[1:]]
        self.target = x0

    def update(self):
        pass
