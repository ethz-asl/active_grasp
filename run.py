import argparse
import rospy

from policies import get_controller


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=str,
        choices=[
            "single-view",
            "fixed-trajectory",
        ],
    )
    return parser


def main():
    rospy.init_node("active_grasp")

    parser = create_parser()
    args = parser.parse_args()

    controller = get_controller(args.policy)
    controller.run()


if __name__ == "__main__":
    main()
