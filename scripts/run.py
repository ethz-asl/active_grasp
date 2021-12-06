#!/usr/bin/env python3

import argparse
from datetime import datetime
import pandas as pd
from pathlib import Path
import rospy
from tqdm import tqdm

from active_grasp.controller import *
from active_grasp.policy import make, registry
from active_grasp.srv import Seed
from robot_helpers.ros import tf


def main():
    rospy.init_node("grasp_controller")
    tf.init()

    parser = create_parser()
    args = parser.parse_args()

    policy = make(args.policy)
    controller = GraspController(policy)
    logger = Logger(args)

    seed_simulation(args.seed)
    rospy.sleep(1.0)  # Prevents a rare race condiion

    for _ in tqdm(range(args.runs)):
        if args.wait_for_input:
            controller.gripper.move(0.08)
            controller.switch_to_joint_trajectory_control()
            controller.moveit.goto("ready")
            i = input("Run policy? [y/n] ")
            if i != "y":
                exit()
            rospy.loginfo("Running policy ...")
        info = controller.run()
        logger.log_run(info)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=str, choices=registry.keys())
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--wait-for-input", action="store_true")
    parser.add_argument("--logdir", type=Path, default="logs")
    parser.add_argument("--seed", type=int, default=1)
    return parser


class Logger:
    def __init__(self, args):
        args.logdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%y%m%d-%H%M%S")
        name = "{}_policy={},seed={}.csv".format(
            stamp,
            args.policy,
            args.seed,
        )
        self.path = args.logdir / name

    def log_run(self, info):
        df = pd.DataFrame.from_records([info])
        df.to_csv(self.path, mode="a", header=not self.path.exists(), index=False)


def seed_simulation(seed):
    rospy.ServiceProxy("seed", Seed)(seed)
    rospy.sleep(1.0)


if __name__ == "__main__":
    main()
