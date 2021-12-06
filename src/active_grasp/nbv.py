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
        self.load_parameters()
        self.compile()

    def load_parameters(self):
        self.min_z_dist = rospy.get_param("~camera/min_z_dist")
        self.max_views = rospy.get_param("nbv_grasp/max_views")
        self.min_gain = rospy.get_param("nbv_grasp/min_gain")
        self.downsample = rospy.get_param("nbv/downsample")

    def compile(self):
        # Trigger the JIT compilation
        raycast(
            1.0,
            np.zeros((40, 40, 40), dtype=np.float32),
            np.eye(3),
            np.zeros(3),
            1.0,
            1.0,
            1.0,
            1.0,
            0,
            1,
            0,
            1,
            0.0,
            1.0,
            0.1,
        )

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
                gains = [self.ig_fn(v, self.downsample) for v in views]
            with Timer("cost_computation"):
                costs = [self.cost_fn(v) for v in views]
            utilities = gains / np.sum(gains) - costs / np.sum(costs)
            self.vis.ig_views(self.base_frame, self.intrinsic, views, utilities)
            i = np.argmax(utilities)
            nbv, gain = views[i], gains[i]

            if gain < self.min_gain and len(self.views) > self.T:
                self.done = True

            self.x_d = nbv

    def best_grasp_prediction_is_stable(self):
        if self.best_grasp:
            t = (self.T_task_base * self.best_grasp.pose).translation
            i, j, k = (t / self.tsdf.voxel_size).astype(int)
            qs = self.qual_hist[:, i, j, k]
            if np.count_nonzero(qs) == self.T and np.mean(qs) > 0.9:
                return True
        return False

    def generate_views(self, q):
        thetas = np.deg2rad([15, 30])
        phis = np.arange(8) * np.deg2rad(45)
        view_candidates = []
        for theta, phi in itertools.product(thetas, phis):
            view = self.view_sphere.get_view(theta, phi)
            if self.solve_cam_ik(q, view):
                view_candidates.append(view)
        return view_candidates

    def ig_fn(self, view, downsample, vis=False):
        tsdf_grid, voxel_size = self.tsdf.get_grid(), self.tsdf.voxel_size
        tsdf_grid = -1.0 + 2.0 * tsdf_grid  # Open3D maps tsdf to [0,1]

        # Downsample the sensor resolution
        fx = self.intrinsic.fx / downsample
        fy = self.intrinsic.fy / downsample
        cx = self.intrinsic.cx / downsample
        cy = self.intrinsic.cy / downsample

        print(fx)

        # Project bbox onto the image plane to get better bounds
        T_cam_base = view.inv()
        corners = np.array([T_cam_base.apply(p) for p in self.bbox.corners]).T
        u = (fx * corners[0] / corners[2] + cx).round().astype(int)
        v = (fy * corners[1] / corners[2] + cy).round().astype(int)
        u_min, u_max = u.min(), u.max()
        v_min, v_max = v.min(), v.max()

        t_min = 0.0  # self.min_z_dist
        t_max = corners[2].max()  # This bound might be a bit too short
        t_step = np.sqrt(3) * voxel_size  # Could be replaced with line rasterization

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

        # Count rear side voxels within the bounding box
        indices = np.unique(voxel_indices, axis=0)
        bbox_min = self.T_task_base.apply(self.bbox.min) / voxel_size
        bbox_max = self.T_task_base.apply(self.bbox.max) / voxel_size
        mask = np.array([((i > bbox_min) & (i < bbox_max)).all() for i in indices])
        i, j, k = indices[mask].T
        tsdfs = tsdf_grid[i, j, k]
        ig = np.logical_and(tsdfs > -1.0, tsdfs < 0.0).sum()

        if vis:
            dirs = []
            for u in range(u_min, u_max):
                for v in range(v_min, v_max):
                    d = np.asarray([(u - cx) / fx, (v - cy) / fy, 1.0])
                    dirs.append(ori @ (d / np.linalg.norm(d)))
            self.vis.rays(self.task_frame, pos, dirs, t_max)

        return ig

    def cost_fn(self, view):
        return 1.0
