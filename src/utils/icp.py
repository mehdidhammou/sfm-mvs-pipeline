import numpy as np
import os
import open3d as o3d
from typing import TypedDict


class IcpConfig(TypedDict):
    source: str
    target: str
    max_iter: int
    tol: float
    std_ratio: float
    nb_neighbors: int
    threshold: float


def read_ply(file_path):
    """Read a PLY file and return a point cloud."""
    return o3d.io.read_point_cloud(file_path)


def write_ply(file_path, pcd):
    """Write a point cloud to a PLY file."""
    o3d.io.write_point_cloud(file_path, pcd)


def scale_point_cloud(pcd, scale):
    """Scale a point cloud by a given factor."""
    points = np.asarray(pcd.points)
    points *= scale
    pcd.points = o3d.utility.Vector3dVector(points)


def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """Remove outliers from the point cloud using statistical outlier removal."""
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return pcd.select_by_index(ind)


def _icp(source_pcd, target_pcd, max_iterations=50, tolerance=1e-6, threshold=1.0):
    """Perform Iterative Closest Point algorithm using Open3D."""
    transformation = np.eye(4)  # Initial transformation matrix

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        threshold,
        transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations, relative_fitness=tolerance
        ),
    )

    return reg_p2p.transformation


def run_icp(args: IcpConfig):
    # Read source and target point clouds
    source_pcd = read_ply(args["source"])
    target_pcd = read_ply(args["target"])

    print("Source point cloud:", source_pcd)
    print("Target point cloud:", target_pcd)

    # Remove outliers
    source_pcd = remove_outliers(source_pcd, args["nb_neighbors"], args["std_ratio"])
    target_pcd = remove_outliers(target_pcd, args["nb_neighbors"], args["std_ratio"])

    print("Source point cloud after removing outliers:", source_pcd)
    print("Target point cloud after removing outliers:", target_pcd)

    # Compute scale factors
    norm_source = np.linalg.norm(np.asarray(source_pcd.points))
    norm_target = np.linalg.norm(np.asarray(target_pcd.points))
    scale_factor = norm_target / norm_source

    # Scale the source point cloud
    scale_point_cloud(source_pcd, scale_factor)

    final_transformation = _icp(
        source_pcd, target_pcd, args["max_iter"], args["tol"], args["threshold"]
    )

    # Apply final transformation to source points
    source_pcd.transform(final_transformation)

    # Merge and save
    output_path = os.path.join("output", "merged.ply")
    write_ply(output_path, source_pcd + target_pcd)

    # Visualize the aligned point clouds
    source_pcd.paint_uniform_color([1, 0, 0])  # Red
    target_pcd.paint_uniform_color([0, 1, 0])  # Green
    o3d.visualization.draw_geometries([source_pcd, target_pcd])
