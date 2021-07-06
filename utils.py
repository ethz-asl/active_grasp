from geometry_msgs.msg import PoseStamped
import rospy

from robot_utils.ros.conversions import *


class CartesianPoseControllerClient:
    def __init__(self, topic="/command"):
        self.target_pub = rospy.Publisher(topic, PoseStamped, queue_size=10)

    def send_target(self, pose):
        msg = to_pose_stamped_msg(pose, "panda_link0")
        self.target_pub.publish(msg)
