import rospy

from active_grasp.srv import *

rospy.init_node("reset")

seed = rospy.ServiceProxy("seed", Seed)
reset = rospy.ServiceProxy("reset", Reset)

seed(SeedRequest(1))

while True:
    reset(ResetRequest())
    rospy.sleep(1.0)
