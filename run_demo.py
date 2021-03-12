import rospy

from controllers import *
from simulation import *
from utils import *


# parameters
gui = True
dt = 1.0 / 60.0

rospy.init_node("demo")

env = SimPandaEnv(gui)
model = env.arm
controller = CartesianPoseController(model)

init_ee_pose = env.arm.pose()
marker = InteractiveMarkerWrapper("target", "panda_link0", init_ee_pose)

# run the control loop
while True:
    controller.set_target(marker.get_pose())
    q, dq = env.arm.get_state()
    cmd = controller.update(q, dq)
    env.arm.set_desired_joint_velocities(cmd)
    env.forward(dt)
