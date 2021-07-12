from datetime import datetime
from geometry_msgs.msg import PoseStamped
import pandas as pd
import rospy
import time

import active_grasp.msg
from robot_utils.ros.conversions import *


class CartesianPoseControllerClient:
    def __init__(self, topic="/command"):
        self.target_pub = rospy.Publisher(topic, PoseStamped, queue_size=10)

    def send_target(self, pose):
        msg = to_pose_stamped_msg(pose, "panda_link0")
        self.target_pub.publish(msg)


class AABBox:
    def __init__(self, bbox_min, bbox_max):
        self.min = bbox_min
        self.max = bbox_max

    def is_inside(self, p):
        return np.all(p > self.min) and np.all(p < self.max)


def from_bbox_msg(msg):
    aabb_min = from_point_msg(msg.min)
    aabb_max = from_point_msg(msg.max)
    return AABBox(aabb_min, aabb_max)


def to_bbox_msg(bbox):
    msg = active_grasp.msg.AABBox()
    msg.min = to_point_msg(bbox.min)
    msg.max = to_point_msg(bbox.max)
    return msg


class Timer:
    timers = dict()

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_info):
        self.stop()

    def start(self):
        self.tic = time.perf_counter()

    def stop(self):
        elapsed_time = time.perf_counter() - self.tic
        self.timers[self.name] = elapsed_time


class Logger:
    def __init__(self, logdir, policy, desc):
        stamp = datetime.now().strftime("%y%m%d-%H%M%S")
        name = "{}_policy={},{}".format(stamp, policy, desc).strip(",")
        self.path = logdir / (name + ".csv")

    def log_run(self, info):
        df = pd.DataFrame.from_records([info])
        df.to_csv(self.path, mode="a", header=not self.path.exists(), index=False)
