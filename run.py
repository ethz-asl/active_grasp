import argparse

from geometry_msgs.msg import Pose
import numpy as np
import rospy
from std_srvs.srv import Trigger

from policies import get_policy


def main(args):
    rospy.init_node("panda_grasp")

    policy = get_policy(args.policy)

    r = rospy.Rate(args.rate)
    done = False
    policy.start()
    while not done:
        done = policy.update()
        r.sleep()

    # TODO execute grasp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, choices=["fixed"])
    parser.add_argument("--rate", type=int, default=10)
    args = parser.parse_args()
    main(args)
