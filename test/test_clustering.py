import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def main():
    cloud_file = "1636465097.pcd"

    # eps, min_points = 0.02, 10
    eps, min_points = 0.01, 8

    cloud = o3d.io.read_point_cloud(cloud_file)

    labels = np.array(cloud.cluster_dbscan(eps=eps, min_points=min_points))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([cloud])


if __name__ == "__main__":
    main()
