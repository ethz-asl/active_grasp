import cv_bridge
import numpy as np
from pathlib import Path
import rospy
from rospy import Publisher
import sensor_msgs.msg
from visualization_msgs.msg import Marker, MarkerArray

from robot_utils.perception import Image
from robot_utils.ros import tf
from robot_utils.ros.conversions import *
from robot_utils.ros.rviz import *
from robot_utils.spatial import Transform
from vgn.detection import VGN, compute_grasps
from vgn.perception import UniformTSDFVolume
from vgn.utils import *


def get_policy(name):
    if name == "single-view":
        return SingleView()
    else:
        raise ValueError("{} policy does not exist.".format(name))


class Policy:
    def activate(self, bbox):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class BasePolicy(Policy):
    def __init__(self):
        self.cv_bridge = cv_bridge.CvBridge()
        self.vgn = VGN(Path(rospy.get_param("vgn/model")))
        self.finger_depth = 0.05

        self.load_parameters()
        self.lookup_transforms()
        self.connect_to_camera()
        self.connect_to_rviz()

        self.rate = 2

    def load_parameters(self):
        self.base_frame = rospy.get_param("~base_frame_id")
        self.task_frame = rospy.get_param("~frame_id")
        self.cam_frame = rospy.get_param("~camera/frame_id")
        self.info_topic = rospy.get_param("~camera/info_topic")
        self.depth_topic = rospy.get_param("~camera/depth_topic")

    def lookup_transforms(self):
        tf._init_listener()
        rospy.sleep(1.0)  # wait to receive transforms
        self.T_B_task = tf.lookup(self.base_frame, self.task_frame)

    def connect_to_camera(self):
        msg = rospy.wait_for_message(
            self.info_topic, sensor_msgs.msg.CameraInfo, rospy.Duration(2.0)
        )
        self.intrinsic = from_camera_info_msg(msg)
        rospy.Subscriber(
            self.depth_topic, sensor_msgs.msg.Image, self.sensor_cb, queue_size=1
        )

    def sensor_cb(self, msg):
        self.img = Image(depth=self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32))
        self.extrinsic = tf.lookup(
            self.cam_frame,
            self.task_frame,
            msg.header.stamp,
            rospy.Duration(0.2),
        )

    def connect_to_rviz(self):
        self.bbox_pub = Publisher("bbox", Marker, queue_size=1, latch=True)
        self.cloud_pub = Publisher("cloud", PointCloud2, queue_size=1, latch=True)
        self.path_pub = Publisher("path", MarkerArray, queue_size=1, latch=True)
        self.grasps_pub = Publisher("grasps", MarkerArray, queue_size=1, latch=True)

    def activate(self, bbox):
        self.clear_grasps()
        self.bbox = bbox
        self.draw_bbox()
        self.tsdf = UniformTSDFVolume(0.3, 40)
        self.viewpoints = []
        self.done = False
        self.best_grasp = None  # grasp pose defined w.r.t. the robot's base frame

    def update(self):
        raise NotImplementedError

    def integrate_latest_image(self):
        self.viewpoints.append(self.extrinsic.inv())
        self.tsdf.integrate(
            self.img,
            self.intrinsic,
            self.extrinsic,
        )

    def predict_best_grasp(self):
        tsdf_grid = self.tsdf.get_grid()
        out = self.vgn.predict(tsdf_grid)
        score_fn = lambda g: g.pose.translation[2]
        grasps = compute_grasps(self.tsdf.voxel_size, out, score_fn, max_filter_size=3)
        grasps = self.filter_grasps_on_target_object(grasps)
        self.draw_grasps(grasps)
        return self.T_B_task * grasps[0].pose if len(grasps) > 0 else None

    def filter_grasps_on_target_object(self, grasps):
        return [
            g
            for g in grasps
            if self.bbox.is_inside(
                g.pose.rotation.apply([0, 0, 0.05]) + g.pose.translation
            )
        ]

    def clear_grasps(self):
        self.grasps_pub.publish(DELETE_MARKER_ARRAY_MSG)

    def draw_bbox(self):
        pose = Transform.translation((self.bbox.min + self.bbox.max) / 2.0)
        scale = self.bbox.max - self.bbox.min
        color = np.r_[0.8, 0.2, 0.2, 0.6]
        msg = create_marker(Marker.CUBE, self.task_frame, pose, scale, color)
        self.bbox_pub.publish(msg)

    def draw_scene_cloud(self):
        cloud = self.tsdf.get_scene_cloud()
        msg = to_cloud_msg(self.task_frame, np.asarray(cloud.points))
        self.cloud_pub.publish(msg)

    def draw_grasps(self, grasps):
        msg = create_grasp_marker_array(self.task_frame, grasps, self.finger_depth)
        self.grasps_pub.publish(msg)

    def draw_camera_path(self):
        identity = Transform.identity()
        color = np.r_[31, 119, 180] / 255.0

        # Spheres for each viewpoint
        scale = 0.01 * np.ones(3)
        spheres = create_marker(
            Marker.SPHERE_LIST, self.task_frame, identity, scale, color
        )
        spheres.id = 0
        spheres.points = [to_point_msg(p.translation) for p in self.viewpoints]

        # Line strip connecting viewpoints
        scale = [0.005, 0.0, 0.0]
        lines = create_marker(
            Marker.LINE_STRIP, self.task_frame, identity, scale, color
        )
        lines.id = 1
        lines.points = [to_point_msg(p.translation) for p in self.viewpoints]

        self.path_pub.publish(MarkerArray([spheres, lines]))


class SingleView(BasePolicy):
    """
    Process a single image from the initial viewpoint.
    """

    def update(self):
        self.integrate_latest_image()
        self.draw_scene_cloud()
        self.best_grasp = self.predict_best_grasp()
        self.done = True
