import numpy as np
import rospy

from robot_helpers.ros import tf


def main():
    rospy.init_node("calibrate_roi")
    tf.init()
    T_base_roi = tf.lookup("panda_link0", "tag_0")
    np.savetxt("cfg/T_base_tag.txt", T_base_roi.as_matrix())


if __name__ == "__main__":
    main()
