from pathlib import Path

import cv_bridge
import numpy as np
import rospy
import scipy.interpolate

from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, CameraInfo

from robot_utils.spatial import Rotation, Transform
from robot_utils.ros.conversions import *
from robot_utils.ros.tf import TransformTree
from robot_utils.perception import *
from vgn import vis
from vgn.detection import VGN, compute_grasps


def get_policy(name):
    if name == "single-view":
        return SingleViewBaseline()
    elif name == "fixed-trajectory":
        return FixedTrajectoryBaseline()
    else:
        raise ValueError("{} policy does not exist.".format(name))


class Policy:
    def __init__(self):
        params = rospy.get_param("active_grasp")

        self.frame_id = params["frame_id"]

        # Robot
        self.tf_tree = TransformTree()
        self.H_EE_G = Transform.from_list(params["ee_grasp_offset"])
        self.target_pose_pub = rospy.Publisher("/target", Pose, queue_size=10)

        # Camera
        camera_name = params["camera_name"]
        self.cam_frame_id = camera_name + "_optical_frame"
        self.cv_bridge = cv_bridge.CvBridge()
        depth_topic = camera_name + "/depth/image_raw"
        rospy.Subscriber(depth_topic, Image, self.sensor_cb, queue_size=1)
        msg = rospy.wait_for_message(camera_name + "/depth/camera_info", CameraInfo)
        self.intrinsic = from_camera_info_msg(msg)

        # VGN
        params = rospy.get_param("vgn")
        self.vgn = VGN(Path(params["model"]))

        rospy.sleep(1.0)
        self.H_B_T = self.tf_tree.lookup("panda_link0", self.frame_id, rospy.Time.now())

    def sensor_cb(self, msg):
        self.last_depth_img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32)
        self.last_extrinsic = self.tf_tree.lookup(
            self.cam_frame_id, self.frame_id, msg.header.stamp, rospy.Duration(0.1)
        )


class SingleViewBaseline(Policy):
    pass


class FixedTrajectoryBaseline(Policy):
    def __init__(self):
        super().__init__()
        self.duration = 4.0
        self.radius = 0.1
        self.m = scipy.interpolate.interp1d([0, self.duration], [np.pi, 3.0 * np.pi])
        self.tsdf = UniformTSDFVolume(0.3, 40)
        vis.draw_workspace(0.3)

    def start(self):
        self.tic = rospy.Time.now()
        timeout = rospy.Duration(0.1)
        x0 = self.tf_tree.lookup("panda_link0", "panda_hand", self.tic, timeout)
        self.origin = np.r_[x0.translation[0] + self.radius, x0.translation[1:]]
        self.target = x0
        self.done = False

    def update(self):
        elapsed_time = (rospy.Time.now() - self.tic).to_sec()

        # Integrate image
        self.tsdf.integrate(
            self.last_depth_img,
            self.intrinsic,
            self.last_extrinsic,
        )

        # Visualize current integration
        cloud = self.tsdf.get_scene_cloud()
        vis.draw_points(np.asarray(cloud.points))

        if elapsed_time > self.duration:
            # Plan grasps
            map_cloud = self.tsdf.get_map_cloud()
            points = np.asarray(map_cloud.points)
            distances = np.asarray(map_cloud.colors)[:, 0]
            tsdf_grid = create_grid_from_map_cloud(
                points, distances, self.tsdf.voxel_size
            )
            out = self.vgn.predict(tsdf_grid)
            grasps = compute_grasps(out, voxel_size=self.tsdf.voxel_size)

            # Visualize
            vis.draw_tsdf(tsdf_grid, self.tsdf.voxel_size)
            vis.draw_grasps(grasps, 0.05)

            # Ensure that the camera is pointing forward.
            grasp = grasps[0]
            rot = grasp.pose.rotation
            axis = rot.as_matrix()[:, 0]
            if axis[0] < 0:
                grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)

            # Compute target pose of the EE
            H_T_G = grasp.pose
            H_B_EE = self.H_B_T * H_T_G * self.H_EE_G.inv()
            self.best_grasp = H_B_EE
            self.done = True
            return

        t = self.m(elapsed_time)
        self.target.translation = (
            self.origin + np.r_[self.radius * np.cos(t), self.radius * np.sin(t), 0.0]
        )
        self.target_pose_pub.publish(to_pose_msg(self.target))
