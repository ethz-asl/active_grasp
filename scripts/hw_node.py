#!/usr/bin/env python3

from controller_manager_msgs.srv import *
import rospy

from active_grasp.bbox import AABBox, to_bbox_msg
from active_grasp.srv import *


from robot_helpers.ros.moveit import MoveItClient


class HwNode:
    def __init__(self):
        self.advertise_services()
        self.switch_controller = rospy.ServiceProxy(
            "controller_manager/switch_controller", SwitchController
        )

        self.moveit = MoveItClient("panda_arm")
        rospy.spin()

    def advertise_services(self):
        rospy.Service("seed", Seed, self.seed)
        rospy.Service("reset", Reset, self.reset)

    def seed(self, req):
        # Nothing to do
        return SeedResponse()

    def reset(self, req):
        req = SwitchControllerRequest()
        req.start_controllers = ["position_joint_trajectory_controller"]
        req.stop_controllers = ["cartesian_velocity_controller"]
        self.switch_controller(req)

        # Move to the initial configuration
        self.moveit.goto([0.0, -0.79, 0.0, -2.356, 0.0, 1.57, 0.79])

        # Detect target
        bbox = AABBox([0.4, -0.1, 0.0], [0.5, 0.1, 0.1])
        return ResetResponse(to_bbox_msg(bbox))


def main():
    rospy.init_node("hw")
    HwNode()


if __name__ == "__main__":
    main()
