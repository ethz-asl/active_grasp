import rospy

from active_grasp.srv import *

rospy.init_node("test")

seed = rospy.ServiceProxy("seed", Seed)
reset = rospy.ServiceProxy("reset", Reset)

# seed(SeedRequest(1))
reset(ResetRequest())
