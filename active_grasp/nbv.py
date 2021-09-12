import itertools
import numpy as np
import rospy
from .timer import Timer

from .policy import MultiViewPolicy


class NextBestView(MultiViewPolicy):
    def __init__(self):
        super().__init__()
        self.min_z_dist = rospy.get_param("~camera/min_z_dist")
        self.max_views = 40

    def activate(self, bbox, view_sphere):
        super().activate(bbox, view_sphere)
        with Timer("view_generation"):
            self.generate_view_candidates()
        self.info["view_candidates_count"] = len(self.view_candidates)

    def generate_view_candidates(self):
        thetas = np.deg2rad([15, 30, 45])
        phis = np.arange(8) * np.deg2rad(45)
        self.view_candidates = []
        for theta, phi in itertools.product(thetas, phis):
            view = self.view_sphere.get_view(theta, phi)
            if self.is_feasible(view):
                self.view_candidates.append(view)

    def update(self, img, x):
        if len(self.views) > self.max_views or self.best_grasp_prediction_is_stable():
            self.done = True
        else:
            with Timer("state_update"):
                self.integrate(img, x)
            views = self.view_candidates
            with Timer("ig_computation"):
                gains = [self.ig_fn(v) for v in views]
            with Timer("cost_computation"):
                costs = [self.cost_fn(v) for v in views]
            utilities = gains / np.sum(gains) - costs / np.sum(costs)
            self.vis.views(self.base_frame, self.intrinsic, views, utilities)
            i = np.argmax(utilities)
            nbv, _ = views[i], gains[i]
            self.x_d = nbv

    def best_grasp_prediction_is_stable(self):
        if self.best_grasp:
            t = (self.T_task_base * self.best_grasp.pose).translation
            i, j, k = (t / self.tsdf.voxel_size).astype(int)
            qs = self.qual_hist[:, i, j, k]
            if (
                np.count_nonzero(qs) == self.T
                and np.mean(qs) > 0.9
                and np.std(qs) < 0.05
            ):
                return True
        return False

    def ig_fn(self, view, downsample=20):
        tsdf_grid, voxel_size = self.tsdf.get_grid(), self.tsdf.voxel_size
        tsdf_grid = -1.0 + 2.0 * tsdf_grid  # Open3D maps tsdf to [0,1]

        fx = self.intrinsic.fx / downsample
        fy = self.intrinsic.fy / downsample
        cx = self.intrinsic.cx / downsample
        cy = self.intrinsic.cy / downsample

        # Project bbox onto the image plane to get better bounds
        T_cam_base = view.inv()
        corners = np.array([T_cam_base.apply(p) for p in self.bbox.corners]).T
        u = (fx * corners[0] / corners[2] + cx).round().astype(int)
        v = (fy * corners[1] / corners[2] + cy).round().astype(int)
        u_min, u_max = u.min(), u.max()
        v_min, v_max = v.min(), v.max()

        t_min = self.min_z_dist
        t_max = corners[2].max()  # TODO This bound might be a bit too short
        t_step = np.sqrt(3) * voxel_size  # TODO replace with line rasterization

        view = self.T_task_base * view  # We'll work in the task frame from now on
        origin = view.translation

        def get_voxel_at(p):
            index = (p / voxel_size).astype(int)
            return index if (index >= 0).all() and (index < 40).all() else None

        voxel_indices = []
        for u in range(u_min, u_max):
            for v in range(v_min, v_max):
                direction = np.r_[(u - cx) / fx, (v - cy) / fy, 1.0]
                direction = view.rotation.apply(direction / np.linalg.norm(direction))
                # self.vis.rays(self.task_frame, origin, [direction])
                t, tsdf_prev = t_min, -1.0
                while t < t_max:
                    p = origin + t * direction
                    t += t_step
                    # self.vis.point(self.task_frame, p)
                    index = get_voxel_at(p)
                    if index is not None:
                        i, j, k = index
                        tsdf = tsdf_grid[i, j, k]
                        if tsdf * tsdf_prev < 0 and tsdf_prev > -1:  # Crossed a surface
                            break
                        voxel_indices.append(index)
                        tsdf_prev = tsdf

        # Count rear side voxels
        i, j, k = np.unique(voxel_indices, axis=0).T
        tsdfs = tsdf_grid[i, j, k]
        ig = np.logical_and(tsdfs > -1.0, tsdfs < 0.0).sum()

        return ig

    def cost_fn(self, view):
        return 1.0
