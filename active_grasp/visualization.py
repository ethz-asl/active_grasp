import numpy as np
import rospy

from robot_helpers.ros.rviz import *
from robot_helpers.spatial import Transform
from vgn.utils import *


class Visualizer:
    def __init__(self, frame, topic="visualization_marker_array"):
        self.frame = frame
        self.marker_pub = rospy.Publisher(topic, MarkerArray, queue_size=1)
        self.scene_cloud_pub = rospy.Publisher("scene_cloud", PointCloud2, queue_size=1)

    def clear(self):
        marker = Marker(action=Marker.DELETEALL)
        self.draw([marker])

    def bbox(self, bbox):
        pose = Transform.translation((bbox.min + bbox.max) / 2.0)
        scale = bbox.max - bbox.min
        color = np.r_[0.8, 0.2, 0.2, 0.6]
        marker = create_cube_marker(self.frame, pose, scale, color, ns="bbox")
        self.draw([marker])

    def scene_cloud(self, frame, cloud):
        msg = to_cloud_msg(frame, np.asarray(cloud.points))
        self.scene_cloud_pub.publish(msg)

    def path(self, poses):
        color = np.r_[31, 119, 180] / 255.0
        points = [p.translation for p in poses]
        spheres = create_sphere_list_marker(
            self.frame,
            Transform.identity(),
            np.full(3, 0.01),
            color,
            points,
            "path",
            0,
        )
        lines = create_line_strip_marker(
            self.frame,
            Transform.identity(),
            [0.005, 0.0, 0.0],
            color,
            points,
            "path",
            1,
        )
        self.draw([spheres, lines])

    def draw(self, markers):
        self.marker_pub.publish(MarkerArray(markers=markers))
