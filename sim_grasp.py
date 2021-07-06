import argparse
import rospy
import std_srvs.srv as std_srvs


from controller import GraspController
from policies import get_policy


class SimGraspController(GraspController):
    def __init__(self, policy):
        super().__init__(policy)
        self.reset_sim = rospy.ServiceProxy("/reset", std_srvs.Trigger)

    def reset(self):
        req = std_srvs.TriggerRequest()
        self.reset_sim(req)
        rospy.sleep(1.0)  # wait for states to be updated


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
    controller = SimGraspController(policy)

    while True:
        controller.run()


if __name__ == "__main__":
    main()
