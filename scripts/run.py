import argparse
import rospy

from active_grasp.controller import GraspController
from active_grasp.policies import get_policy


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=str,
        choices=[
            "single-view",
            "top",
            "alignment",
            "random",
            "fixed-trajectory",
        ],
    )
    return parser


def main():
    rospy.init_node("active_grasp")
    parser = create_parser()
    args = parser.parse_args()
    policy = get_policy(args.policy)
    controller = GraspController(policy)

    while True:
        controller.run()


if __name__ == "__main__":
    main()
