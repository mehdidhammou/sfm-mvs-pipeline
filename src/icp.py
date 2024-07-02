import numpy as np
import sys
import open3d as o3d
import argparse


def read_ply(file_path):
    """Read a PLY file and return a point cloud."""
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd


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


def icp(source_pcd, target_pcd, max_iterations=50, tolerance=1e-6, threshold=1.0):
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


def main(args):
    # Read source and target point clouds
    source_pcd = read_ply(args.source)
    target_pcd = read_ply(args.target)

    print("Source point cloud:", source_pcd)
    print("Target point cloud:", target_pcd)

    # Remove outliers
    source_pcd = remove_outliers(source_pcd, args.nb_neighbors, args.std_ratio)
    target_pcd = remove_outliers(target_pcd, args.nb_neighbors, args.std_ratio)

    print("Source point cloud after removing outliers:", source_pcd)
    print("Target point cloud after removing outliers:", target_pcd)

    # Compute scale factors
    norm_source = np.linalg.norm(np.asarray(source_pcd.points))
    norm_target = np.linalg.norm(np.asarray(target_pcd.points))
    scale_factor = norm_target / norm_source

    # Scale the source point cloud
    scale_point_cloud(source_pcd, scale_factor)

    # Run ICP algorithm
    final_transformation = icp(
        source_pcd, target_pcd, args.max_iter, args.tol, args.threshold
    )

    # Apply final transformation to source points
    source_pcd.transform(final_transformation)

    # Merge the transformed source point cloud with the target point cloud
    merged_pcd = source_pcd + target_pcd

    # Write merged point clouds to a new PLY file
    write_ply("merged.ply", merged_pcd)

    # Optionally, visualize the aligned point clouds
    source_pcd.paint_uniform_color([1, 0, 0])  # Red
    target_pcd.paint_uniform_color([0, 1, 0])  # Green
    o3d.visualization.draw_geometries([source_pcd, target_pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align two point clouds using ICP with optional outlier removal and scaling."
    )
    parser.add_argument("source", type=str, help="Path to the source PLY file.")
    parser.add_argument("target", type=str, help="Path to the target PLY file.")
    parser.add_argument(
        "-i",
        "--max_iter",
        type=int,
        default=50,
        help="Maximum number of ICP iterations.",
    )
    parser.add_argument(
        "-t", "--tol", type=float, default=1e-6, help="Tolerance for ICP convergence."
    )
    parser.add_argument(
        "-s",
        "--std_ratio",
        type=float,
        default=2.0,
        help="Standard deviation ratio for statistical outlier removal.",
    )
    parser.add_argument(
        "-n",
        "--nb_neighbors",
        type=int,
        default=20,
        help="Number of neighbors to analyze for each point in statistical outlier removal.",
    )
    parser.add_argument(
        "-d", "--threshold", type=float, default=1.0, help="Distance threshold for ICP."
    )

    args = parser.parse_args()
    main(args)
