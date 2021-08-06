import argparse
from datetime import datetime
import pandas as pd
from pathlib import Path
import rospy
from tqdm import tqdm

from active_grasp.controller import *
from active_grasp.policy import make, registry
from active_grasp.srv import Seed


def main():
    rospy.init_node("active_grasp")

    parser = create_parser()
    args = parser.parse_args()

    policy = make(args.policy, args.rate)
    controller = GraspController(policy)
    logger = Logger(args)

    seed_simulation(args.seed)

    for _ in tqdm(range(args.runs)):
        info = controller.run()
        logger.log_run(info)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=str, choices=registry.keys())
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--logdir", type=Path, default="logs")
    parser.add_argument("--rate", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12)
    return parser


class Logger:
    def __init__(self, args):
        stamp = datetime.now().strftime("%y%m%d-%H%M%S")
        descr = "policy={},rate={}".format(args.policy, args.rate)
        self.path = args.logdir / (stamp + "_" + descr + ".csv")

    def log_run(self, info):
        df = pd.DataFrame.from_records([info])
        df.to_csv(self.path, mode="a", header=not self.path.exists(), index=False)


def seed_simulation(seed):
    rospy.ServiceProxy("seed", Seed)(seed)
    rospy.sleep(1.0)


if __name__ == "__main__":
    main()
