import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import open3d as o3d

from src.icp import (
    merge_point_clouds,
    read_ply,
    remove_outliers,
    scale_point_cloud,
    write_ply,
)


class TestMergePointClouds(unittest.TestCase):

    def setUp(self):
        """Set up two sample point clouds for testing."""
        # Create mock point cloud data for source and target
        source_points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        source_colors = np.array([[0.5, 0.5, 0.5], [0.6, 0.7, 0.8]])

        target_points = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        target_colors = np.array([[0.9, 0.8, 0.7], [0.1, 0.2, 0.3]])

        # Create Open3D PointCloud objects
        self.source_pcd = o3d.geometry.PointCloud()
        self.source_pcd.points = o3d.utility.Vector3dVector(source_points)
        self.source_pcd.colors = o3d.utility.Vector3dVector(source_colors)

        self.target_pcd = o3d.geometry.PointCloud()
        self.target_pcd.points = o3d.utility.Vector3dVector(target_points)
        self.target_pcd.colors = o3d.utility.Vector3dVector(target_colors)

    def test_merge_point_clouds(self):
        """Test that the merge_point_clouds function correctly merges point clouds."""
        merged_pcd = merge_point_clouds(self.source_pcd, self.target_pcd)

        # Check merged points and colors
        merged_points = np.asarray(merged_pcd.points)
        merged_colors = np.asarray(merged_pcd.colors)

        expected_points = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        )
        expected_colors = np.array(
            [[0.5, 0.5, 0.5], [0.6, 0.7, 0.8], [0.9, 0.8, 0.7], [0.1, 0.2, 0.3]]
        )

        # Assert points and colors match the expected result
        np.testing.assert_array_equal(merged_points, expected_points)
        np.testing.assert_array_equal(merged_colors, expected_colors)


class TestPointCloudFunctions(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and a sample point cloud."""
        # Create a temporary directory to store test files
        self.test_dir = TemporaryDirectory()
        self.ply_file = os.path.join(self.test_dir.name, "test.ply")

        # Create a simple point cloud for testing
        self.pcd = o3d.geometry.PointCloud()
        points = np.random.rand(100, 3)  # 100 random 3D points
        colors = np.random.rand(100, 3)  # 100 random RGB colors
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        # Write the sample point cloud to a PLY file
        write_ply(self.ply_file, self.pcd)

    def test_read_ply(self):
        """Test reading a PLY file."""
        pcd = read_ply(self.ply_file)
        self.assertTrue(pcd.has_points())  # Ensure points were read
        self.assertEqual(np.asarray(pcd.points).shape[0], 100)  # Check point count

    def test_write_ply(self):
        """Test writing a point cloud to a PLY file."""
        new_ply_file = os.path.join(self.test_dir.name, "new_test.ply")
        write_ply(new_ply_file, self.pcd)
        pcd = read_ply(new_ply_file)
        self.assertTrue(pcd.has_points())
        self.assertEqual(np.asarray(pcd.points).shape[0], 100)

    def test_scale_point_cloud(self):
        """Test scaling a point cloud."""
        original_points = np.asarray(self.pcd.points).copy()
        scale = 2.0
        scale_point_cloud(self.pcd, scale)
        scaled_points = np.asarray(self.pcd.points)
        self.assertTrue(np.allclose(scaled_points, original_points * scale))

    def test_remove_outliers(self):
        """Test removing outliers from the point cloud."""
        original_point_count = len(np.asarray(self.pcd.points))
        pcd_no_outliers = remove_outliers(self.pcd, nb_neighbors=20, std_ratio=2.0)
        new_point_count = len(np.asarray(pcd_no_outliers.points))
        self.assertLess(
            new_point_count, original_point_count
        )  # Points should be reduced

    def tearDown(self):
        """Cleanup temporary directory."""
        self.test_dir.cleanup()
