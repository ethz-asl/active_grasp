from controller_manager_msgs.srv import *
import copy
import cv_bridge
from geometry_msgs.msg import Twist
import numpy as np
import rospy
from sensor_msgs.msg import Image

from .bbox import from_bbox_msg
from .timer import Timer
from active_grasp.srv import Reset, ResetRequest
from robot_helpers.ros import tf
from robot_helpers.ros.conversions import *
from robot_helpers.ros.panda import PandaArmClient, PandaGripperClient
from robot_helpers.ros.moveit import MoveItClient
from robot_helpers.spatial import Rotation, Transform
from vgn.utils import look_at, cartesian_to_spherical, spherical_to_cartesian


class GraspController:
    def __init__(self, policy):
        self.policy = policy
        self.load_parameters()
        self.init_service_proxies()
        self.init_robot_connection()
        self.init_moveit()
        self.init_camera_stream()

    def load_parameters(self):
        self.base_frame = rospy.get_param("~base_frame_id")
        self.T_grasp_ee = Transform.from_list(rospy.get_param("~ee_grasp_offset")).inv()
        self.cam_frame = rospy.get_param("~camera/frame_id")
        self.depth_topic = rospy.get_param("~camera/depth_topic")
        self.min_z_dist = rospy.get_param("~camera/min_z_dist")
        self.control_rate = rospy.get_param("~control_rate")
        self.linear_vel = rospy.get_param("~linear_vel")
        self.policy_rate = rospy.get_param("~policy_rate")

    def init_service_proxies(self):
        self.reset_env = rospy.ServiceProxy("reset", Reset)
        self.switch_controller = rospy.ServiceProxy(
            "controller_manager/switch_controller", SwitchController
        )

    def init_robot_connection(self):
        self.arm = PandaArmClient()
        self.gripper = PandaGripperClient()
        self.cartesian_vel_pub = rospy.Publisher("command", Twist, queue_size=10)

    def init_moveit(self):
        self.moveit = MoveItClient("panda_arm")
        rospy.sleep(1.0)  # Wait for connections to be established.
        # msg = to_pose_stamped_msg(Transform.t([0.4, 0, 0.4]), self.base_frame)
        # self.moveit.scene.add_box("table", msg, size=(0.5, 0.5, 0.02))

    def switch_to_cartesian_velocity_control(self):
        req = SwitchControllerRequest()
        req.start_controllers = ["cartesian_velocity_controller"]
        req.stop_controllers = ["position_joint_trajectory_controller"]
        self.switch_controller(req)

    def switch_to_joint_trajectory_control(self):
        req = SwitchControllerRequest()
        req.start_controllers = ["position_joint_trajectory_controller"]
        req.stop_controllers = ["cartesian_velocity_controller"]
        self.switch_controller(req)

    def init_camera_stream(self):
        self.cv_bridge = cv_bridge.CvBridge()
        rospy.Subscriber(self.depth_topic, Image, self.sensor_cb, queue_size=1)

    def sensor_cb(self, msg):
        self.latest_depth_msg = msg

    def run(self):
        bbox = self.reset()

        self.switch_to_cartesian_velocity_control()
        with Timer("search_time"):
            grasp = self.search_grasp(bbox)

        self.switch_to_joint_trajectory_control()
        with Timer("grasp_time"):
            res = self.execute_grasp(grasp)

        return self.collect_info(res)

    def reset(self):
        Timer.reset()
        res = self.reset_env(ResetRequest())
        rospy.sleep(1.0)  # Wait for the TF tree to be updated.
        return from_bbox_msg(res.bbox)

    def search_grasp(self, bbox):
        self.view_sphere = ViewHalfSphere(bbox, self.min_z_dist)
        self.policy.activate(bbox, self.view_sphere)
        timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.send_vel_cmd)
        r = rospy.Rate(self.policy_rate)
        while not self.policy.done:
            img, pose, q = self.get_state()
            self.policy.update(img, pose, q)
            r.sleep()
        rospy.sleep(0.1)  # Wait for a zero command to be sent to the robot.
        timer.shutdown()
        return self.policy.best_grasp

    def get_state(self):
        q, _ = self.arm.get_state()
        msg = copy.deepcopy(self.latest_depth_msg)
        img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32)
        pose = tf.lookup(self.base_frame, self.cam_frame, msg.header.stamp)
        return img, pose, q

    def send_vel_cmd(self, event):
        if self.policy.x_d is None or self.policy.done:
            cmd = np.zeros(6)
        else:
            x = tf.lookup(self.base_frame, self.cam_frame)
            cmd = self.compute_velocity_cmd(self.policy.x_d, x)
        self.cartesian_vel_pub.publish(to_twist_msg(cmd))

    def compute_velocity_cmd(self, x_d, x):
        r, theta, phi = cartesian_to_spherical(x.translation - self.view_sphere.center)
        e_t = x_d.translation - x.translation
        e_n = (x.translation - self.view_sphere.center) * (self.view_sphere.r - r) / r
        linear = 1.0 * e_t + 6.0 * (r < self.view_sphere.r) * e_n
        scale = np.linalg.norm(linear)
        linear *= np.clip(scale, 0.0, self.linear_vel) / scale
        angular = self.view_sphere.get_view(theta, phi).rotation * x.rotation.inv()
        angular = 0.5 * angular.as_rotvec()
        return np.r_[linear, angular]

    def execute_grasp(self, grasp):
        if not grasp:
            return "aborted"
        T_base_grasp = self.postprocess(grasp.pose)
        self.gripper.move(0.08)
        self.moveit.goto(T_base_grasp * Transform.t([0, 0, -0.05]) * self.T_grasp_ee)
        self.moveit.goto(T_base_grasp * self.T_grasp_ee)
        self.gripper.grasp()
        self.moveit.goto(Transform.t([0, 0, 0.1]) * T_base_grasp * self.T_grasp_ee)
        success = self.gripper.read() > 0.005
        return "succeeded" if success else "failed"

    def postprocess(self, T_base_grasp):
        rot = T_base_grasp.rotation
        if rot.as_matrix()[:, 0][0] < 0:  # Ensure that the camera is pointing forward
            T_base_grasp.rotation = rot * Rotation.from_euler("z", np.pi)
        T_base_grasp *= Transform.t([0.0, 0.0, 0.01])
        return T_base_grasp

    def collect_info(self, result):
        points = [p.translation for p in self.policy.views]
        d = np.sum([np.linalg.norm(p2 - p1) for p1, p2 in zip(points, points[1:])])
        info = {
            "result": result,
            "view_count": len(points),
            "distance": d,
        }
        info.update(self.policy.info)
        info.update(Timer.timers)
        return info


class ViewHalfSphere:
    def __init__(self, bbox, min_z_dist):
        self.center = bbox.center
        self.r = 0.5 * bbox.size[2] + min_z_dist

    def get_view(self, theta, phi):
        eye = self.center + spherical_to_cartesian(self.r, theta, phi)
        up = np.r_[1.0, 0.0, 0.0]
        return look_at(eye, self.center, up)

    def sample_view(self):
        raise NotImplementedError
