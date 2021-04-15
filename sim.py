import argparse

import numpy as np
import rospy

from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from robot_tools.btsim import BtPandaEnv
from robot_tools.controllers import CartesianPoseController
from robot_tools.ros import *

CONTROLLER_UPDATE_RATE = 60
JOINT_STATE_PUBLISHER_RATE = 60


class BtSimNode:
    def __init__(self, gui):
        self.sim = BtPandaEnv(gui=gui, sleep=False)
        self.controller = CartesianPoseController(self.sim.arm)

        self.joint_state_pub = rospy.Publisher(
            "/joint_states", JointState, queue_size=10
        )
        rospy.Subscriber("/target", Pose, self._target_pose_cb)

    def run(self):
        rate = rospy.Rate(self.sim.rate)
        self.step_cnt = 0
        while not rospy.is_shutdown():
            self._handle_updates()
            self.sim.step()
            self.step_cnt = (self.step_cnt + 1) % self.sim.rate
            rate.sleep()

    def _handle_updates(self):
        if self.step_cnt % int(self.sim.rate / CONTROLLER_UPDATE_RATE) == 0:
            self.controller.update()
        if self.step_cnt % int(self.sim.rate / JOINT_STATE_PUBLISHER_RATE) == 0:
            self._publish_joint_state()

    def _publish_joint_state(self):
        q, dq = self.sim.arm.get_state()
        width = self.sim.gripper.read()
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ["panda_joint{}".format(i) for i in range(1, 8)] + [
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]
        msg.position = np.r_[q, 0.5 * width, 0.5 * width]
        msg.velocity = dq
        self.joint_state_pub.publish(msg)

    def _target_pose_cb(self, msg):
        self.controller.set_target(from_pose_msg(msg))


def main(args):
    rospy.init_node("bt_sim")
    server = BtSimNode(args.gui)
    server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=str, default=True)
    args = parser.parse_args()
    main(args)
