from geometry_msgs.msg import Pose
import numpy as np
import rospy
import scipy.interpolate

from robot_utils.spatial import Rotation, Transform
from robot_utils.ros.conversions import *
from robot_utils.ros.tf import TransformTree


def get_policy(name):
    if name == "fixed-trajectory":
        return FixedTrajectory()
    else:
        raise ValueError("{} policy does not exist.".format(name))


class BasePolicy:
    def __init__(self):
        self.tf_tree = TransformTree()
        self.target_pose_pub = rospy.Publisher("/target", Pose, queue_size=10)
        rospy.sleep(1.0)


class FixedTrajectory(BasePolicy):
    def __init__(self):
        super().__init__()
        self.duration = 4.0
        self.radius = 0.1
        self.m = scipy.interpolate.interp1d([0, self.duration], [np.pi, 3.0 * np.pi])

    def start(self):
        self.tic = rospy.Time.now()
        timeout = rospy.Duration(0.1)
        x0 = self.tf_tree.lookup("panda_link0", "panda_hand", self.tic, timeout)
        self.origin = np.r_[x0.translation[0] + self.radius, x0.translation[1:]]
        self.target = x0

    def update(self):
        elapsed_time = (rospy.Time.now() - self.tic).to_sec()

        if elapsed_time > self.duration:
            return True

        t = self.m(elapsed_time)
        self.target.translation = (
            self.origin + np.r_[self.radius * np.cos(t), self.radius * np.sin(t), 0.0]
        )
        self.target_pose_pub.publish(to_pose_msg(self.target))
        return False
