import argparse
from pathlib import Path
import rospy
from tqdm import tqdm

from active_grasp.controller import *
from active_grasp.policy import make, registry


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=str, choices=registry.keys())
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--logdir", type=Path, default="logs")
    parser.add_argument("--desc", type=str, default="")
    return parser


def main():
    rospy.init_node("active_grasp")
    parser = create_parser()
    args = parser.parse_args()
    policy = make(args.policy)
    controller = GraspController(policy)
    logger = Logger(args.logdir, args.policy, args.desc)

    for _ in tqdm(range(args.runs)):
        info = controller.run()
        logger.log_run(info)


if __name__ == "__main__":
    main()
