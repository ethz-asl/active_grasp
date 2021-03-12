import numpy as np
from scipy.spatial.transform import Rotation
import PyKDL as kdl

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *


class InteractiveMarkerWrapper(object):
    def __init__(self, name, frame_id, pose):
        server = InteractiveMarkerServer(topic_ns=name)

        target_marker = InteractiveMarker()
        target_marker.header.frame_id = frame_id
        target_marker.name = name
        target_marker.scale = 0.2
        target_marker.pose.position.x = pose.translation[0]
        target_marker.pose.position.y = pose.translation[1]
        target_marker.pose.position.z = pose.translation[2]

        marker = Marker()
        marker.type = Marker.SPHERE
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 0.6
        ctrl = InteractiveMarkerControl()
        ctrl.always_visible = True
        ctrl.markers.append(marker)
        target_marker.controls.append(ctrl)

        ctrl = InteractiveMarkerControl()
        ctrl.name = "move_x"
        ctrl.orientation.w = 1.0
        ctrl.orientation.x = 1.0
        ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        target_marker.controls.append(ctrl)

        ctrl = InteractiveMarkerControl()
        ctrl.name = "move_y"
        ctrl.orientation.w = 1.0
        ctrl.orientation.y = 1.0
        ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        target_marker.controls.append(ctrl)

        ctrl = InteractiveMarkerControl()
        ctrl.name = "move_z"
        ctrl.orientation.w = 1.0
        ctrl.orientation.z = 1.0
        ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        target_marker.controls.append(ctrl)

        server.insert(target_marker, self.cb)
        server.applyChanges()

        self.pose = pose

    def cb(self, feedback):
        pos = feedback.pose.position
        self.pose = Transform(Rotation.identity(), np.r_[pos.x, pos.y, pos.z])

    def get_pose(self):
        return self.pose


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
