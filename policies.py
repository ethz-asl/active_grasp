import cv_bridge
import numpy as np
from pathlib import Path
import rospy
import sensor_msgs.msg
from visualization_msgs.msg import Marker, MarkerArray

from robot_utils.perception import Image
from robot_utils.ros import tf
from robot_utils.ros.conversions import *
from robot_utils.ros.rviz import *
from robot_utils.spatial import Transform
from vgn.detection import VGN, compute_grasps
from vgn.perception import UniformTSDFVolume
import vgn.vis


def get_policy(name):
    if name == "single-view":
        return SingleViewBaseline()
    else:
        raise ValueError("{} policy does not exist.".format(name))


class BasePolicy:
    def __init__(self):
        self.cv_bridge = cv_bridge.CvBridge()
        self.tsdf = UniformTSDFVolume(0.3, 40)
        self.vgn = VGN(Path(rospy.get_param("vgn/model")))

        self.load_parameters()
        self.lookup_transforms()
        self.connect_to_camera()
        self.connect_to_rviz()

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
        self.path_pub = rospy.Publisher("path", MarkerArray, queue_size=1, latch=True)

    def activate(self):
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
        grasps = compute_grasps(self.tsdf.voxel_size, out, score_fn)
        vgn.vis.draw_grasps(grasps, 0.05)
        return self.T_B_task * grasps[0].pose if len(grasps) > 0 else None

    def draw_scene_cloud(self):
        cloud = self.tsdf.get_scene_cloud()
        vgn.vis.draw_points(np.asarray(cloud.points))

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


class SingleViewBaseline(BasePolicy):
    """
    Process a single image from the initial viewpoint.
    """

    def __init__(self):
        super().__init__()
        self.rate = 1

    def update(self):
        self.integrate_latest_image()
        self.draw_scene_cloud()
        self.draw_camera_path()
        self.best_grasp = self.predict_best_grasp()
        self.done = True
