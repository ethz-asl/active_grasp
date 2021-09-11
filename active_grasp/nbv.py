import itertools
import numpy as np
import rospy

from .policy import MultiViewPolicy, compute_error
from vgn.utils import look_at


class NextBestView(MultiViewPolicy):
    def __init__(self, rate):
        super().__init__(rate)
        self.max_views = 20
        self.min_ig = 10.0
        self.cost_factor = 10.0

    def activate(self, bbox):
        super().activate(bbox)
        self.generate_view_candidates()

    def update(self, img, x):
        if len(self.views) > self.max_views:
            self.done = True
            return np.zeros(6)
        else:
            self.integrate(img, x)
            views = self.view_candidates
            gains = self.compute_expected_information_gains(views)
            costs = self.compute_movement_costs(views)
            utilities = gains / np.sum(gains) - costs / np.sum(costs)
            self.vis.views(self.base_frame, self.intrinsic, views, utilities)
            i = np.argmax(utilities)
            nbv, ig = views[i], gains[i]
            cmd = self.compute_velocity_cmd(*compute_error(nbv, x))
            if self.best_grasp:
                R, t = self.best_grasp.pose.rotation, self.best_grasp.pose.translation
                if np.linalg.norm(t - x.translation) < self.min_z_dist:
                    self.done = True
                    return np.zeros(6)

                center = t
                eye = R.apply([0.0, 0.0, -0.2]) + t
                up = np.r_[1.0, 0.0, 0.0]
                x_d = look_at(eye, center, up)
                cmd = self.compute_velocity_cmd(*compute_error(x_d, x))
            return cmd

    def generate_view_candidates(self):
        thetas = np.arange(1, 4) * np.deg2rad(30)
        phis = np.arange(1, 6) * np.deg2rad(60)
        self.view_candidates = []
        for theta, phi in itertools.product(thetas, phis):
            view = self.view_sphere.get_view(theta, phi)
            if self.is_view_feasible(view):
                self.view_candidates.append(view)

    def compute_expected_information_gains(self, views):
        return [self.ig_fn(v) for v in views]

    def ig_fn(self, view, downsample=20):
        fx = self.intrinsic.fx / downsample
        fy = self.intrinsic.fy / downsample
        cx = self.intrinsic.cx / downsample
        cy = self.intrinsic.cy / downsample

        T_cam_base = view.inv()
        corners = np.array([T_cam_base.apply(p) for p in self.bbox.corners]).T
        u = (fx * corners[0] / corners[2] + cx).round().astype(int)
        v = (fy * corners[1] / corners[2] + cy).round().astype(int)
        u_min, u_max = u.min(), u.max()
        v_min, v_max = v.min(), v.max()

        t_min = 0.1
        t_max = corners[2].max()  # TODO This bound might be a bit too short
        t_step = 0.01

        view = self.T_task_base * view  # We'll work in the task frame from now on

        tsdf_grid, voxel_size = self.tsdf.get_grid(), self.tsdf.voxel_size

        def get_voxel_at(p):
            index = (p / voxel_size).astype(int)
            return index if (index >= 0).all() and (index < 40).all() else None

        voxel_indices = []

        for u in range(u_min, u_max):
            for v in range(v_min, v_max):
                origin = view.translation
                direction = np.r_[(u - cx) / fx, (v - cy) / fy, 1.0]
                direction = view.rotation.apply(direction / np.linalg.norm(direction))

                # self.vis.rays(self.task_frame, origin, [direction])
                # rospy.sleep(0.01)

                t, tsdf_prev = t_min, -1.0
                while t < t_max:
                    p = origin + t * direction
                    t += t_step

                    # self.vis.point(self.task_frame, p)
                    # rospy.sleep(0.01)

                    index = get_voxel_at(p)
                    if index is not None:
                        i, j, k = index
                        tsdf = -1 + 2 * tsdf_grid[i, j, k]  # Open3D maps tsdf to [0,1]
                        if tsdf * tsdf_prev < 0 and tsdf_prev > -1:  # Crossed a surface
                            break
                        # TODO check whether the voxel lies within the bounding box ?
                        voxel_indices.append(index)
                        tsdf_prev = tsdf

        # Count rear side voxels
        i, j, k = np.unique(voxel_indices, axis=0).T
        tsdfs = tsdf_grid[i, j, k]
        ig = np.logical_and(tsdfs > 0.0, tsdfs < 0.5).sum()

        return ig

    def compute_movement_costs(self, views):
        return [self.cost_fn(v) for v in views]

    def cost_fn(self, view):
        return 1.0
