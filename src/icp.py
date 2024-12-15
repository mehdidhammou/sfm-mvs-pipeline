import argparse
import os
from glob import glob
from typing import TypedDict

import numpy as np
import open3d as o3d


class IcpConfig(TypedDict):
    source: str
    target: str
    output_dir: str
    max_iter: int
    tol: float
    std_ratio: float
    nb_neighbors: int
    threshold: float


def get_argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align multiple point clouds using ICP with optional outlier removal and scaling."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the directory containing point cloud PLY files.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the directory to save the merged point cloud.",
    )
    parser.add_argument(
        "-i",
        "--max_iter",
        type=int,
        default=50,
        help="Maximum number of ICP iterations.",
    )
    parser.add_argument(
        "-t",
        "--tol",
        type=float,
        default=1e-6,
        help="Tolerance for ICP convergence.",
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

    return parser.parse_args()


def merge_point_clouds(source_pcd, target_pcd):
    """Merge two point clouds including positions and colors."""
    # Extend positions and colors
    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)

    source_colors = np.asarray(source_pcd.colors)
    target_colors = np.asarray(target_pcd.colors)

    # Concatenate points and colors
    merged_points = np.vstack([source_points, target_points])
    merged_colors = np.vstack([source_colors, target_colors])

    # Create a new point cloud with merged data
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
    merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

    return merged_pcd


def draw_geometry(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Merged Point Clouds", width=800, height=600)
    vis.add_geometry(pcd)

    # Optionally, adjust the viewpoint
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])  # Set the viewpoint front direction
    view_control.set_up([0, -1, 0])  # Set the viewpoint up direction

    vis.run()
    vis.destroy_window()


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


import os
import open3d as o3d
import numpy as np


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

    # Perform ICP
    final_transformation = _icp(
        source_pcd, target_pcd, args["max_iter"], args["tol"], args["threshold"]
    )

    # Apply final transformation to source points
    source_pcd.transform(final_transformation)

    # Merge source and target point clouds
    merged_pcd = merge_point_clouds(source_pcd, target_pcd)

    # Ensure output directory exists
    os.makedirs(args["output_dir"], exist_ok=True)

    # Save the merged point cloud to a PLY file
    output_path = os.path.join(args["output_dir"], "merged.ply")
    write_ply(output_path, merged_pcd)

    print(f"Successfully saved merged point cloud to {output_path}")

    # Visualize the aligned point clouds
    draw_geometry(merged_pcd)


if __name__ == "__main__":
    args = get_argparser()
    # Collect all PLY files in the input directory
    ply_files = sorted(glob(os.path.join(args.input_dir, "*.ply")))
    if len(ply_files) < 2:
        raise ValueError("The input directory must contain at least two PLY files.")

    # Initialize source with the first PLY file
    source = ply_files[0]
    os.makedirs(args.output_dir, exist_ok=True)

    # Iteratively align and merge point clouds
    for target in ply_files[1:]:
        print(f"Merging {source} with {target}...")
        params = {
            "source": source,
            "target": target,
            "output_dir": args.output_dir,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "std_ratio": args.std_ratio,
            "nb_neighbors": args.nb_neighbors,
            "threshold": args.threshold,
        }
        run_icp(params)
        # Save intermediate merged result as the new source
        source = os.path.join(args.output_dir, "merged.ply")

    print(f"Final merged point cloud saved to {args.output_dir}.")
