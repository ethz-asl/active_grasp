import rospy

from controllers import *
from robot_sim import *
from utils import *


gui = True

rospy.init_node("demo")

env = SimPandaEnv(gui)

q, dq = env.arm.get_state()
x0 = env.model.pose(q)

env.controller = CartesianPoseController(env.arm, env.model, x0, 60)
marker = InteractiveMarkerWrapper("target", "panda_link0", x0)

while True:
    env.controller.set_target(marker.pose)
    env.camera.update()
    env.forward(0.1)
