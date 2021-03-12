import rospy

from controllers import *
from model import *
from robot_sim import *
from utils import *


# parameters
gui = True
dt = 1.0 / 60.0

rospy.init_node("demo")

env = SimPandaEnv(gui)
model = Model("panda_link0", "panda_link8")
controller = CartesianPoseController(model)

q, dq = env.arm.get_state()
marker = InteractiveMarkerWrapper("target", "panda_link0", model.pose(q))

# run the control loop
while True:
    controller.set_target(marker.get_pose())
    q, dq = env.arm.get_state()
    cmd = controller.update(q, dq)
    env.arm.set_desired_joint_velocities(cmd)

    env.forward(dt)
