import itertools
from numba import jit
import numpy as np
import rospy

from .policy import MultiViewPolicy

from .timer import Timer


@jit(nopython=True)
def get_voxel_at(voxel_size, p):
    index = (p / voxel_size).astype(np.int64)
    return index if (index >= 0).all() and (index < 40).all() else None


# Note that the jit compilation takes some time the first time raycast is called
@jit(nopython=True)
def raycast(
    voxel_size,
    tsdf_grid,
    ori,
    pos,
    fx,
    fy,
    cx,
    cy,
    u_min,
    u_max,
    v_min,
    v_max,
    t_min,
    t_max,
    t_step,
):
    voxel_indices = []
    for u in range(u_min, u_max):
        for v in range(v_min, v_max):
            direction = np.asarray([(u - cx) / fx, (v - cy) / fy, 1.0])
            direction = ori @ (direction / np.linalg.norm(direction))
            t, tsdf_prev = t_min, -1.0
            while t < t_max:
                p = pos + t * direction
                t += t_step
                index = get_voxel_at(voxel_size, p)
                if index is not None:
                    i, j, k = index
                    tsdf = tsdf_grid[i, j, k]
                    if tsdf * tsdf_prev < 0 and tsdf_prev > -1:  # crossed a surface
                        break
                    voxel_indices.append(index)
                    tsdf_prev = tsdf
    return voxel_indices


class NextBestView(MultiViewPolicy):
    def __init__(self):
        super().__init__()
        self.min_z_dist = rospy.get_param("~camera/min_z_dist")
        self.max_views = 50
        self.min_gain = 0.0

    def activate(self, bbox, view_sphere):
        super().activate(bbox, view_sphere)

    def update(self, img, x, q):
        if len(self.views) > self.max_views or self.best_grasp_prediction_is_stable():
            self.done = True
        else:
            with Timer("state_update"):
                self.integrate(img, x, q)
            with Timer("view_generation"):
                views = self.generate_views(q)
            with Timer("ig_computation"):
                gains = [self.ig_fn(v, 10) for v in views]
            with Timer("cost_computation"):
                costs = [self.cost_fn(v) for v in views]
            utilities = gains / np.sum(gains) - costs / np.sum(costs)
            self.vis.views(self.base_frame, self.intrinsic, views, utilities)
            i = np.argmax(utilities)
            nbv, gain = views[i], gains[i]

            if gain < self.min_gain:
                self.done = True

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

    def generate_views(self, q):
        thetas = np.deg2rad([15, 30, 45])
        phis = np.arange(6) * np.deg2rad(60)
        view_candidates = []
        for theta, phi in itertools.product(thetas, phis):
            view = self.view_sphere.get_view(theta, phi)
            if self.is_feasible(view, q):
                view_candidates.append(view)
        return view_candidates

    def ig_fn(self, view, downsample):
        tsdf_grid, voxel_size = self.tsdf.get_grid(), self.tsdf.voxel_size
        tsdf_grid = -1.0 + 2.0 * tsdf_grid  # Open3D maps tsdf to [0,1]

        # Downsample the sensor resolution
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

        # Cast rays from the camera view (we'll work in the task frame from now on)
        view = self.T_task_base * view
        ori, pos = view.rotation.as_matrix(), view.translation

        voxel_indices = raycast(
            voxel_size,
            tsdf_grid,
            ori,
            pos,
            fx,
            fy,
            cx,
            cy,
            u_min,
            u_max,
            v_min,
            v_max,
            t_min,
            t_max,
            t_step,
        )

        # Count rear side voxels
        i, j, k = np.unique(voxel_indices, axis=0).T
        tsdfs = tsdf_grid[i, j, k]
        ig = np.logical_and(tsdfs > -1.0, tsdfs < 0.0).sum()

        return ig

    def cost_fn(self, view):
        return 1.0
