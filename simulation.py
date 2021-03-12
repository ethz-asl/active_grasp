import time

import numpy as np
import pybullet as p
import pybullet_data

from sensor_msgs.msg import JointState

from utils import *


class PandaArm(object):
    def __init__(self):
        self.num_dof = 7
        self.ee_link_id = 7
        self.ee_frame_id = "panda_link7"

        self.named_joint_values = {"ready": [0.0, -0.79, 0.0, -2.356, 0.0, 1.57, 0.79]}

        self.upper_limits = [-7] * self.num_dof
        self.lower_limits = [7] * self.num_dof
        self.ranges = [7] * self.num_dof

    def load(self, pose):
        self.T_world_base = pose
        self.uid = p.loadURDF(
            "assets/urdfs/panda/panda_arm_hand.urdf",
            basePosition=pose.translation,
            baseOrientation=pose.rotation.as_quat(),
            useFixedBase=True,
        )
        for i in range(self.num_dof):
            p.resetJointState(self.uid, i, self.named_joint_values["ready"][i])

    def get_state(self):
        joint_states = p.getJointStates(self.uid, range(p.getNumJoints(self.uid)))[:7]
        positions = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]
        return positions, velocities

    def set_desired_joint_positions(self, positions):
        for i, position in enumerate(positions):
            p.setJointMotorControl2(self.uid, i, p.POSITION_CONTROL, position)

    def set_desired_joint_velocities(self, velocities):
        for i, velocity in enumerate(velocities):
            p.setJointMotorControl2(
                self.uid, i, p.VELOCITY_CONTROL, targetVelocity=velocity
            )

    def pose(self):
        result = p.getLinkState(self.uid, self.ee_link_id, computeForwardKinematics=1)
        _, _, _, _, frame_pos, frame_quat = result
        T_world_ee = Transform(Rotation.from_quat(frame_quat), np.array(frame_pos))
        return self.T_world_base.inverse() * T_world_ee

    def jacobian(self, q):
        q = np.r_[q, 0.0, 0.0].tolist()
        q_dot, q_ddot = np.zeros(9).tolist(), np.zeros(9).tolist()
        linear, _ = p.calculateJacobian(self.uid, 7, [0.0, 0.0, 0.0], q, q_dot, q_ddot)
        return np.asarray(linear)[:, :7]


class PandaGripper(object):
    def move(self, width):
        for i in [9, 10]:
            p.setJointMotorControl2(
                self.uid,
                i,
                p.POSITION_CONTROL,
                0.5 * width,
                force=10,
            )

    def read(self):
        return p.getJointState(self.uid, 9)[0] + p.getJointState(self.uid, 10)[0]


class Camera(object):
    pass


class SimPandaEnv(object):
    def __init__(self, gui, dt=1.0 / 240.0, publish_state=True):
        self.gui = gui
        self.dt = dt
        p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(dt)
        p.setGravity(0.0, 0.0, -9.8)

        p.loadURDF("plane.urdf")
        p.loadURDF(
            "table/table.urdf",
            baseOrientation=Rotation.from_rotvec(np.array([0, 0, np.pi / 2])).as_quat(),
            useFixedBase=True,
        )
        p.loadURDF("cube_small.urdf", [0.0, 0.0, 0.8])

        self.arm = PandaArm()
        self.arm.load(Transform(Rotation.identity(), np.r_[-0.8, 0.0, 0.4]))
        self.gripper = PandaGripper()
        self.gripper.uid = self.arm.uid

        self.camera = None

        if publish_state:
            self.state_pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
            rospy.Timer(rospy.Duration(1.0 / 30), self._publish_state)

    def forward(self, dt):
        steps = int(dt / self.dt)
        for _ in range(steps):
            self.step()

    def step(self):
        p.stepSimulation()
        time.sleep(self.dt)

    def _publish_state(self, event):
        positions, velocities = self.arm.get_state()
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ["panda_joint{}".format(i) for i in range(1, 8)]
        msg.position = positions
        msg.velocity = velocities
        self.state_pub.publish(msg)
