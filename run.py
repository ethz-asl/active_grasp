import argparse

from geometry_msgs.msg import Pose
import numpy as np
import rospy
from std_srvs.srv import Trigger

from policies import get_policy

from robot_tools.ros import *


class GraspController:
    def __init__(self, policy, rate):
        self.policy = policy
        self.rate = rate

        self.target_pose_pub = rospy.Publisher("/target", Pose, queue_size=10)
        self.gripper = PandaGripperRosInterface()

    def explore(self):
        r = rospy.Rate(self.rate)
        done = False
        self.policy.start()
        while not done:
            done = self.policy.update()
            r.sleep()

    def execute_grasp(self):
        self.gripper.move(0.08)
        rospy.sleep(1.0)
        target = self.policy.best_grasp
        self.target_pose_pub.publish(to_pose_msg(target))
        rospy.sleep(2.0)
        self.gripper.move(0.0)
        rospy.sleep(1.0)
        target.translation[2] += 0.1
        self.target_pose_pub.publish(to_pose_msg(target))
        rospy.sleep(2.0)


def main(args):
    rospy.init_node("panda_grasp")

    policy = get_policy(args.policy)
    gc = GraspController(policy, args.rate)
    gc.explore()
    gc.execute_grasp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy", type=str, choices=["single-view", "fixed-trajectory"]
    )
    parser.add_argument("--rate", type=int, default=10)
    args = parser.parse_args()
    main(args)
