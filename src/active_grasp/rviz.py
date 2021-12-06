import numpy as np

from robot_helpers.ros.rviz import *
from robot_helpers.spatial import Transform
import vgn.rviz
from vgn.utils import *


cm = lambda s: tuple([float(1 - s), float(s), float(0)])
red = [1.0, 0.0, 0.0]
blue = [0, 0.6, 1.0]
grey = [0.9, 0.9, 0.9]


class Visualizer(vgn.rviz.Visualizer):
    def bbox(self, frame, bbox):
        pose = Transform.identity()
        scale = [0.004, 0.0, 0.0]
        color = red
        lines = box_lines(bbox.min, bbox.max)
        marker = create_line_list_marker(frame, pose, scale, color, lines, "bbox")
        self.draw([marker])

    def ig_views(self, frame, intrinsic, views, values):
        vmin, vmax = min(values), max(values)
        scale = [0.002, 0.0, 0.0]
        near, far = 0.0, 0.02
        markers = []
        for i, (view, value) in enumerate(zip(views, values)):
            color = cm((value - vmin) / (vmax - vmin))
            marker = create_view_marker(
                frame,
                view,
                scale,
                color,
                intrinsic,
                near,
                far,
                ns="ig_views",
                id=i,
            )
            markers.append(marker)
        self.draw(markers)

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
        markers = [spheres]
        if len(poses) > 1:
            lines = create_line_strip_marker(
                frame,
                Transform.identity(),
                [0.005, 0.0, 0.0],
                color,
                points,
                "path",
                1,
            )
            markers.append(lines)
        self.draw(markers)

    def point(self, frame, point):
        marker = create_sphere_marker(
            frame,
            Transform.translation(point),
            np.full(3, 0.01),
            [0, 0, 1],
            "point",
        )
        self.draw([marker])

    def rays(self, frame, origin, directions, t_max=1.0):
        lines = [[origin, origin + t_max * direction] for direction in directions]
        marker = create_line_list_marker(
            frame,
            Transform.identity(),
            [0.001, 0.0, 0.0],
            grey,
            lines,
            "rays",
        )
        self.draw([marker])

    def views(self, frame, intrinsic, views):
        markers = []
        for i, view in enumerate(views):
            markers.append(
                create_view_marker(
                    frame,
                    view,
                    [0.002, 0.0, 0.0],
                    blue,
                    intrinsic,
                    0.0,
                    0.02,
                    ns="views",
                    id=i,
                )
            )
        self.draw(markers)


def create_view_marker(frame, pose, scale, color, intrinsic, near, far, ns="", id=0):
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
