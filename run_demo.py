import rospy

from controllers import *
from robot_sim import *
from utils import *


gui = True

rospy.init_node("demo")

env = SimPandaEnv(gui)
env.controller = CartesianPoseController(env.arm, env.model, 60)

q, dq = env.arm.get_state()
x0 = env.model.pose(q)
marker = InteractiveMarkerWrapper("target", "panda_link0", x0)

while True:
    env.controller.set_target(marker.pose)
    env.forward(0.1)
