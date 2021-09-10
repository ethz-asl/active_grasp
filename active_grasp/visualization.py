from geometry_msgs.msg import PoseArray
import matplotlib.colors
import numpy as np
import rospy

from robot_helpers.ros.rviz import *
from robot_helpers.spatial import Transform
from vgn.utils import *

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("RedGreen", ["r", "g"])
red = np.r_[1.0, 0.0, 0.0]
blue = np.r_[0, 0.6, 1.0]


class Visualizer:
    def __init__(self, topic="visualization_marker_array"):
        self.marker_pub = rospy.Publisher(topic, MarkerArray, queue_size=1)
        self.scene_cloud_pub = rospy.Publisher(
            "scene_cloud",
            PointCloud2,
            latch=True,
            queue_size=1,
        )
        self.map_cloud_pub = rospy.Publisher(
            "map_cloud",
            PointCloud2,
            latch=True,
            queue_size=1,
        )
        self.quality_pub = rospy.Publisher("quality", PointCloud2, queue_size=1)

    def clear(self):
        self.clear_markers()
        msg = to_cloud_msg("panda_link0", np.array([]))
        self.scene_cloud_pub.publish(msg)
        self.map_cloud_pub.publish(msg)
        self.quality_pub.publish(msg)
        rospy.sleep(0.1)

    def clear_markers(self):
        self.draw([Marker(action=Marker.DELETEALL)])

    def clear_views(self, n):
        markers = [Marker(action=Marker.DELETE, ns="views", id=i) for i in range(n)]
        self.draw(markers)

    def draw(self, markers):
        self.marker_pub.publish(MarkerArray(markers=markers))

    def bbox(self, frame, bbox):
        pose = Transform.identity()
        scale = [0.004, 0.0, 0.0]
        color = red
        corners = bbox.corners
        edges = [
            (0, 1),
            (1, 3),
            (3, 2),
            (2, 0),
            (4, 5),
            (5, 7),
            (7, 6),
            (6, 4),
            (0, 4),
            (1, 5),
            (3, 7),
            (2, 6),
        ]
        lines = [(corners[s], corners[e]) for s, e in edges]
        marker = create_line_list_marker(frame, pose, scale, color, lines, ns="bbox")
        self.draw([marker])

    def best_grasp(self, frame, grasp, score, smin=0.9, smax=1.0, alpha=1.0):
        color = cmap((score - smin) / (smax - smin))
        color = [color[0], color[1], color[2], alpha]
        self.draw(create_grasp_markers(frame, grasp, color, "best_grasp", radius=0.006))

    def grasps(self, frame, grasps, scores, smin=0.9, smax=1.0, alpha=0.8):
        if len(grasps) == 0:
            return

        markers = []
        for i, (grasp, score) in enumerate(zip(grasps, scores)):
            color = cmap((score - smin) / (smax - smin))
            color = [color[0], color[1], color[2], alpha]
            markers += create_grasp_markers(frame, grasp, color, "grasps", 4 * i)
        self.draw(markers)

    def rays(self, frame, origin, directions, t_max=1.0):
        lines = [[origin, origin + t_max * direction] for direction in directions]
        marker = create_line_list_marker(
            frame,
            Transform.identity(),
            [0.002, 0.0, 0.0],
            [0.6, 0.6, 0.6],
            lines,
            "rays",
        )
        self.draw([marker])

    def map_cloud(self, frame, cloud):
        points = np.asarray(cloud.points)
        distances = np.expand_dims(np.asarray(cloud.colors)[:, 0], 1)
        msg = to_cloud_msg(frame, points, distances=distances)
        self.map_cloud_pub.publish(msg)

    def path(self, frame, poses):
        color = blue
        points = [p.translation for p in poses]
        spheres = create_sphere_list_marker(
            frame,
            Transform.identity(),
            np.full(3, 0.01),
            color,
            points,
            "path",
            0,
        )
        lines = create_line_strip_marker(
            frame,
            Transform.identity(),
            [0.005, 0.0, 0.0],
            color,
            points,
            "path",
            1,
        )
        self.draw([spheres, lines])

    def point(self, frame, point):
        marker = create_sphere_marker(
            frame,
            Transform.translation(point),
            np.full(3, 0.01),
            [0, 0, 1],
            "point",
        )
        self.draw([marker])

    def quality(self, frame, voxel_size, quality, threshold=0.9):
        points, values = grid_to_map_cloud(voxel_size, quality, threshold)
        values = (values - 0.9) / (1.0 - 0.9)  # to increase contrast
        msg = to_cloud_msg(frame, points, intensities=values)
        self.quality_pub.publish(msg)

    def scene_cloud(self, frame, cloud):
        msg = to_cloud_msg(frame, np.asarray(cloud.points))
        self.scene_cloud_pub.publish(msg)

    def views(self, frame, intrinsic, views, values, alpha=0.8):
        vmin, vmax = min(values), max(values)
        scale = [0.002, 0.0, 0.0]
        near, far = 0.0, 0.02
        markers = []
        for i, (view, value) in enumerate(zip(views, values)):
            color = cmap((value - vmin) / (vmax - vmin))
            color = [color[0], color[1], color[2], alpha]
            marker = create_cam_view_marker(
                frame,
                view,
                scale,
                color,
                intrinsic,
                near,
                far,
                ns="views",
                id=i,
            )
            markers.append(marker)
        self.draw(markers)

    def workspace(self, frame, size):
        scale = size * 0.005
        pose = Transform.identity()
        scale = [scale, 0.0, 0.0]
        color = [0.5, 0.5, 0.5]
        lines = [
            ([0.0, 0.0, 0.0], [size, 0.0, 0.0]),
            ([size, 0.0, 0.0], [size, size, 0.0]),
            ([size, size, 0.0], [0.0, size, 0.0]),
            ([0.0, size, 0.0], [0.0, 0.0, 0.0]),
            ([0.0, 0.0, size], [size, 0.0, size]),
            ([size, 0.0, size], [size, size, size]),
            ([size, size, size], [0.0, size, size]),
            ([0.0, size, size], [0.0, 0.0, size]),
            ([0.0, 0.0, 0.0], [0.0, 0.0, size]),
            ([size, 0.0, 0.0], [size, 0.0, size]),
            ([size, size, 0.0], [size, size, size]),
            ([0.0, size, 0.0], [0.0, size, size]),
        ]
        msg = create_line_list_marker(frame, pose, scale, color, lines, ns="workspace")
        self.draw([msg])


def create_cam_view_marker(
    frame, pose, scale, color, intrinsic, near, far, ns="", id=0
):
    marker = create_marker(Marker.LINE_LIST, frame, pose, scale, color, ns, id)
    x_n = near * intrinsic.width / (2.0 * intrinsic.fx)
    y_n = near * intrinsic.height / (2.0 * intrinsic.fy)
    z_n = near
    x_f = far * intrinsic.width / (2.0 * intrinsic.fx)
    y_f = far * intrinsic.height / (2.0 * intrinsic.fy)
    z_f = far
    points = [
        [x_n, y_n, z_n],
        [-x_n, y_n, z_n],
        [-x_n, y_n, z_n],
        [-x_n, -y_n, z_n],
        [-x_n, -y_n, z_n],
        [x_n, -y_n, z_n],
        [x_n, -y_n, z_n],
        [x_n, y_n, z_n],
        [x_f, y_f, z_f],
        [-x_f, y_f, z_f],
        [-x_f, y_f, z_f],
        [-x_f, -y_f, z_f],
        [-x_f, -y_f, z_f],
        [x_f, -y_f, z_f],
        [x_f, -y_f, z_f],
        [x_f, y_f, z_f],
        [x_n, y_n, z_n],
        [x_f, y_f, z_f],
        [-x_n, y_n, z_n],
        [-x_f, y_f, z_f],
        [-x_n, -y_n, z_n],
        [-x_f, -y_f, z_f],
        [x_n, -y_n, z_n],
        [x_f, -y_f, z_f],
    ]
    marker.points = [to_point_msg(p) for p in points]
    return marker


def create_grasp_markers(
    frame,
    grasp,
    color,
    ns,
    id=0,
    finger_depth=0.05,
    radius=0.003,
):
    w, d = grasp.width, finger_depth

    pose = grasp.pose * Transform.translation([0.0, -w / 2, d / 2])
    scale = [radius, radius, d]
    left = create_marker(Marker.CYLINDER, frame, pose, scale, color, ns, id)

    pose = grasp.pose * Transform.translation([0.0, w / 2, d / 2])
    scale = [radius, radius, d]
    right = create_marker(Marker.CYLINDER, frame, pose, scale, color, ns, id + 1)

    pose = grasp.pose * Transform.translation([0.0, 0.0, -d / 4])
    scale = [radius, radius, d / 2]
    wrist = create_marker(Marker.CYLINDER, frame, pose, scale, color, ns, id + 2)

    pose = grasp.pose * Transform.rotation(Rotation.from_rotvec([np.pi / 2, 0, 0]))
    scale = [radius, radius, w]
    palm = create_marker(Marker.CYLINDER, frame, pose, scale, color, ns, id + 3)

    return [left, right, wrist, palm]
