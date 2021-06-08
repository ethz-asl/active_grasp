import cv_bridge
import numpy as np
from pathlib import Path
import rospy
import scipy.interpolate

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
import std_srvs.srv
from visualization_msgs.msg import Marker, MarkerArray


from robot_utils.perception import *
from robot_utils.spatial import Rotation, Transform
from robot_utils.ros.conversions import *
from robot_utils.ros.panda import PandaGripperClient
from robot_utils.ros import tf
from robot_utils.ros.rviz import *
import vgn.vis

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
        self.frame = rospy.get_param("~frame_id")
        self.length = rospy.get_param("~length")

        self.cv_bridge = cv_bridge.CvBridge()
        self.reset_client = rospy.ServiceProxy("/reset", std_srvs.srv.Trigger)

        self.tsdf = UniformTSDFVolume(0.3, 40)
        self.vgn = VGN(Path(rospy.get_param("vgn/model")))

        self._setup_robot_connection()
        self._setup_camera_connection()
        self._setup_rviz_connection()
        self._lookup_transforms()

    def run(self):
        self.reset()
        self.explore()
        self.execute_grasp()

    def reset(self):
        self._reset_env()
        self._clear_rviz()
        rospy.sleep(1.0)  # wait for states to be updated
        self._init_policy()

        self.viewpoints = []
        self.done = False
        self.best_grasp = None

    def explore(self):
        r = rospy.Rate(self.rate)
        while not self.done:
            self._update()
            r.sleep()

    def execute_grasp(self):
        if not self.best_grasp:
            return

        grasp = self.best_grasp

        # Ensure that the camera is pointing forward.
        rot = grasp.pose.rotation
        if rot.as_matrix()[:, 0][0] < 0:
            grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)
        target = self.T_B_O * grasp.pose * self._ee_grasp_offset.inv()

        self.gripper.move(0.08)
        self._send_pose_command(target)
        rospy.sleep(3.0)
        self.gripper.move(0.0)
        target.translation[2] += 0.3
        self._send_pose_command(target)
        rospy.sleep(2.0)

    def _setup_robot_connection(self):
        self.base_frame = rospy.get_param("~base_frame_id")
        self.ee_frame = rospy.get_param("~ee_frame_id")
        self._ee_grasp_offset = Transform.from_list(rospy.get_param("~ee_grasp_offset"))
        self.target_pose_pub = rospy.Publisher("/command", PoseStamped, queue_size=10)
        self.gripper = PandaGripperClient()

    def _setup_camera_connection(self):
        self._cam_frame_id = rospy.get_param("~camera/frame_id")
        info_topic = rospy.get_param("~camera/info_topic")
        msg = rospy.wait_for_message(info_topic, CameraInfo, rospy.Duration(2.0))
        self.intrinsic = from_camera_info_msg(msg)
        depth_topic = rospy.get_param("~camera/depth_topic")
        rospy.Subscriber(depth_topic, Image, self._sensor_cb, queue_size=1)

    def _sensor_cb(self, msg):
        self.last_depth_img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32)
        self.last_extrinsic = tf.lookup(
            self._cam_frame_id,
            self.frame,
            msg.header.stamp,
            rospy.Duration(0.1),
        )

    def _setup_rviz_connection(self):
        self.path_pub = rospy.Publisher("path", MarkerArray, queue_size=1, latch=True)

    def _lookup_transforms(self):
        self.T_B_O = tf.lookup(self.base_frame, self.frame)

    def _reset_env(self):
        req = std_srvs.srv.TriggerRequest()
        self.reset_client(req)

    def _clear_rviz(self):
        vgn.vis.clear()
        self.path_pub.publish(DELETE_MARKER_ARRAY_MSG)

    def _init_policy(self):
        raise NotImplementedError

    def _update(self):
        raise NotImplementedError

    def _draw_camera_path(self, frustum=False):
        identity = Transform.identity()
        color = np.r_[31, 119, 180] / 255.0

        # Spheres for each viewpoint
        scale = 0.01 * np.ones(3)
        spheres = create_marker(Marker.SPHERE_LIST, self.frame, identity, scale, color)
        spheres.id = 0
        spheres.points = [to_point_msg(p.translation) for p in self.viewpoints]

        # Line strip connecting viewpoints
        scale = [0.005, 0.0, 0.0]
        lines = create_marker(Marker.LINE_STRIP, self.frame, identity, scale, color)
        lines.id = 1
        lines.points = [to_point_msg(p.translation) for p in self.viewpoints]

        markers = [spheres, lines]

        # Frustums
        if frustum:
            for i, pose in enumerate(self.viewpoints):
                msg = create_cam_marker(self.intrinsic, pose, self.frame)
                msg.id = i + 2
                markers.append(msg)

        self.path_pub.publish(MarkerArray(markers))

    def _draw_scene_cloud(self):
        cloud = self.tsdf.get_scene_cloud()
        vgn.vis.draw_points(np.asarray(cloud.points))

    def _integrate_latest_image(self):
        self.viewpoints.append(self.last_extrinsic.inv())
        self.tsdf.integrate(
            self.last_depth_img,
            self.intrinsic,
            self.last_extrinsic,
        )

    def _predict_best_grasp(self):
        tsdf_grid = self.tsdf.get_grid()
        out = self.vgn.predict(tsdf_grid)
        score_fn = lambda g: g.pose.translation[2]
        grasps = compute_grasps(self.tsdf.voxel_size, out, score_fn)
        vgn.vis.draw_grasps(grasps, 0.05)
        return grasps[0] if len(grasps) > 0 else None

    def _send_pose_command(self, target):
        msg = PoseStamped()
        msg.header.frame_id = self.base_frame
        msg.pose = to_pose_msg(target)
        self.target_pose_pub.publish(msg)


class SingleViewBaseline(BaseController):
    """
    Integrate a single image from the initial viewpoint.
    """

    def __init__(self):
        super().__init__()
        self.rate = 1

    def _init_policy(self):
        pass

    def _update(self):
        self._integrate_latest_image()
        self._draw_scene_cloud()
        self._draw_camera_path(frustum=True)
        self.best_grasp = self._predict_best_grasp()
        self.done = True


class FixedTrajectoryBaseline(BaseController):
    """Follow a pre-defined circular trajectory."""

    def __init__(self):
        super().__init__()
        self.rate = 10
        self.duration = 4.0
        self.radius = 0.1
        self.m = scipy.interpolate.interp1d([0, self.duration], [np.pi, 3.0 * np.pi])

    def _init_policy(self):
        self.tic = rospy.Time.now()
        x0 = tf.lookup(self.base_frame, self.ee_frame)
        self.center = np.r_[x0.translation[0] + self.radius, x0.translation[1:]]
        self.target = x0

    def _update(self):
        elapsed_time = (rospy.Time.now() - self.tic).to_sec()
        if elapsed_time > self.duration:
            self.best_grasp = self._predict_best_grasp()
            self.done = True
        else:
            # Update state
            self._integrate_latest_image()

            # Compute next viewpoint
            t = self.m(elapsed_time)
            self.target.translation = (
                self.center
                + np.r_[self.radius * np.cos(t), self.radius * np.sin(t), 0.0]
            )
            self._send_pose_command(self.target)

            # Draw
            self._draw_scene_cloud()
            self._draw_camera_path()


class MultiViewPicking(BaseController):
    pass
