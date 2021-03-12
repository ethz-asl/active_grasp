import numpy as np
import kdl_parser_py.urdf as kdl_parser
import PyKDL as kdl

import utils


class Model(object):
    def __init__(self, root_frame_id, tip_frame_id):
        _, tree = kdl_parser.treeFromFile("assets/urdfs/panda/panda_arm_hand.urdf")
        chain = tree.getChain(root_frame_id, tip_frame_id)
        self.nr_joints = chain.getNrOfJoints()
        self.fk_pos_solver = kdl.ChainFkSolverPos_recursive(chain)
        self.fk_vel_solver = kdl.ChainFkSolverVel_recursive(chain)
        self.jac_solver = kdl.ChainJntToJacSolver(chain)
        return

    def pose(self, q):
        jnt_array = utils.to_kdl_jnt_array(q)
        frame = kdl.Frame()
        self.fk_pos_solver.JntToCart(jnt_array, frame)
        return utils.Transform.from_kdl(frame)

    def velocities(self, q, dq):
        jnt_array_vel = kdl.JntArrayVel(
            utils.to_kdl_jnt_array(q), utils.to_kdl_jnt_array(dq)
        )
        twist = kdl.FrameVel()
        self.fk_vel_solver.JntToCart(jnt_array_vel, twist)
        d = twist.deriv()
        linear, angular = np.r_[d[0], d[1], d[2]], np.r_[d[3], d[4], d[5]]
        return linear, angular

    def jacobian(self, q):
        jnt_array = utils.to_kdl_jnt_array(q)
        J = kdl.Jacobian(self.nr_joints)
        self.jac_solver.JntToJac(jnt_array, J)
        return utils.kdl_to_mat(J)
