import numpy as np
from scipy.spatial.transform import Rotation
import PyKDL as kdl

import geometry_msgs.msg
from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
import std_msgs.msg
from visualization_msgs.msg import *


class InteractiveMarkerWrapper(object):
    def __init__(self, name, frame_id, x0):
        self.pose = x0

        server = InteractiveMarkerServer(topic_ns=name)

        int_marker = InteractiveMarker()
        int_marker.header.frame_id = frame_id
        int_marker.name = name
        int_marker.scale = 0.2
        int_marker.pose = to_pose_msg(x0)

        # Attach visible sphere
        marker = Marker()
        marker.type = Marker.SPHERE
        marker.scale = to_vector3_msg([0.05, 0.05, 0.05])
        marker.color = to_color_msg([0.0, 0.5, 0.5, 0.6])

        ctrl = InteractiveMarkerControl()
        ctrl.always_visible = True
        ctrl.markers.append(marker)
        int_marker.controls.append(ctrl)

        # Attach rotation controls
        ctrl = InteractiveMarkerControl()
        ctrl.name = "rotate_x"
        ctrl.orientation = to_quat_msg(Rotation.from_quat([1, 0, 0, 1]))
        ctrl.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(ctrl)

        ctrl = InteractiveMarkerControl()
        ctrl.name = "rotate_y"
        ctrl.orientation = to_quat_msg(Rotation.from_quat([0, 1, 0, 1]))
        ctrl.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(ctrl)

        ctrl = InteractiveMarkerControl()
        ctrl.name = "rotate_z"
        ctrl.orientation = to_quat_msg(Rotation.from_quat([0, 0, 1, 1]))
        ctrl.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(ctrl)

        # Attach translation controls
        ctrl = InteractiveMarkerControl()
        ctrl.name = "move_x"
        ctrl.orientation = to_quat_msg(Rotation.from_quat([1, 0, 0, 1]))
        ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(ctrl)

        ctrl = InteractiveMarkerControl()
        ctrl.name = "move_y"
        ctrl.orientation = to_quat_msg(Rotation.from_quat([0, 1, 0, 1]))
        ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(ctrl)

        ctrl = InteractiveMarkerControl()
        ctrl.name = "move_z"
        ctrl.orientation = to_quat_msg(Rotation.from_quat([0, 0, 1, 1]))
        ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(ctrl)

        server.insert(int_marker, self.cb)
        server.applyChanges()

    def cb(self, feedback):
        self.pose = from_pose_msg(feedback.pose)


class Transform(object):
    def __init__(self, rotation, translation):
        assert isinstance(rotation, Rotation)
        assert isinstance(translation, (np.ndarray, list))

        self.rotation = rotation
        self.translation = np.asarray(translation, np.double)

    def as_matrix(self):
        return np.vstack(
            (np.c_[self.rotation.as_matrix(), self.translation], [0.0, 0.0, 0.0, 1.0])
        )

    def to_list(self):
        return np.r_[self.rotation.as_quat(), self.translation]

    def __mul__(self, other):
        rotation = self.rotation * other.rotation
        translation = self.rotation.apply(other.translation) + self.translation
        return self.__class__(rotation, translation)

    def inverse(self):
        rotation = self.rotation.inv()
        translation = -rotation.apply(self.translation)
        return self.__class__(rotation, translation)

    @classmethod
    def identity(cls):
        rotation = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
        translation = np.array([0.0, 0.0, 0.0])
        return cls(rotation, translation)

    @classmethod
    def from_matrix(cls, m):
        rotation = Rotation.from_matrix(m[:3, :3])
        translation = m[:3, 3]
        return cls(rotation, translation)

    @classmethod
    def from_list(cls, list):
        rotation = Rotation.from_quat(list[:4])
        translation = list[4:]
        return cls(rotation, translation)

    @classmethod
    def from_kdl(cls, f):
        rotation = Rotation.from_matrix(
            np.array(
                [
                    [f.M[0, 0], f.M[0, 1], f.M[0, 2]],
                    [f.M[1, 0], f.M[1, 1], f.M[1, 2]],
                    [f.M[2, 0], f.M[2, 1], f.M[2, 2]],
                ]
            )
        )
        translation = np.r_[f.p[0], f.p[1], f.p[2]]
        return cls(rotation, translation)


# KDL Conversions


def to_kdl_jnt_array(q):
    jnt_array = kdl.JntArray(len(q))
    for i, q_i in enumerate(q):
        jnt_array[i] = q_i
    return jnt_array


def kdl_to_mat(m):
    mat = np.zeros((m.rows(), m.columns()))
    for i in range(m.rows()):
        for j in range(m.columns()):
            mat[i, j] = m[i, j]
    return mat


# ROS Conversions


def to_color_msg(color):
    msg = std_msgs.msg.ColorRGBA()
    msg.r = color[0]
    msg.g = color[1]
    msg.b = color[2]
    msg.a = color[3]
    return msg


def to_point_msg(point):
    msg = geometry_msgs.msg.Point()
    msg.x = point[0]
    msg.y = point[1]
    msg.z = point[2]
    return msg


def from_point_msg(msg):
    return np.r_[msg.x, msg.y, msg.z]


def to_vector3_msg(vector3):
    msg = geometry_msgs.msg.Vector3()
    msg.x = vector3[0]
    msg.y = vector3[1]
    msg.z = vector3[2]
    return msg


def from_vector3_msg(msg):
    return np.r_[msg.x, msg.y, msg.z]


def to_quat_msg(orientation):
    quat = orientation.as_quat()
    msg = geometry_msgs.msg.Quaternion()
    msg.x = quat[0]
    msg.y = quat[1]
    msg.z = quat[2]
    msg.w = quat[3]
    return msg


def from_quat_msg(msg):
    return Rotation.from_quat([msg.x, msg.y, msg.z, msg.w])


def to_pose_msg(transform):
    msg = geometry_msgs.msg.Pose()
    msg.position = to_point_msg(transform.translation)
    msg.orientation = to_quat_msg(transform.rotation)
    return msg


def from_pose_msg(msg):
    position = from_point_msg(msg.position)
    orientation = from_quat_msg(msg.orientation)
    return Transform(orientation, position)