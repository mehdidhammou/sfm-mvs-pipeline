import unittest
import os
import numpy as np
from unittest.mock import patch, MagicMock
from src.utils import Sfm


class TestSfmToPly(unittest.TestCase):

    @patch("os.makedirs")
    @patch("numpy.savetxt")
    def test_to_ply_success(self, mock_savetxt, mock_makedirs):
        # Mock data
        output_dir = "output_dir"
        point_cloud = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])

        sfm_instance = Sfm()

        # Mock the sub_dir part
        with patch.object(sfm_instance, "img_loader", new=MagicMock()):
            sfm_instance.img_loader.image_list = ["path/to/image"]

            # Call the method
            sfm_instance.to_ply(output_dir, point_cloud, colors)

        # Check that os.makedirs was called to create the directory
        mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)

        # Check that np.savetxt was called to write to the file
        mock_savetxt.assert_called_once()

        # Validate the file write call
        args, kwargs = mock_savetxt.call_args
        self.assertEqual(
            args[0].name, "output_dir/path/to"
        )  # Replace with correct path
        # Ensure the right format and content is passed
        self.assertTrue(np.array_equal(args[1], np.hstack([point_cloud * 200, colors])))

    @patch("os.makedirs")
    @patch("numpy.savetxt")
    def test_to_ply_invalid_point_cloud(self, mock_savetxt, mock_makedirs):
        # Invalid point cloud (incorrect shape)
        output_dir = "output_dir"
        point_cloud = np.array([[1, 2], [4, 5], [7, 8]])  # Invalid shape
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])

        sfm_instance = Sfm()

        # Mock the sub_dir part
        with patch.object(sfm_instance, "img_loader", new=MagicMock()):
            sfm_instance.img_loader.image_list = ["path/to/image"]

            # Call the method and expect it to raise an error
            with self.assertRaises(ValueError):
                sfm_instance.to_ply(output_dir, point_cloud, colors)

    @patch("os.makedirs")
    @patch("numpy.savetxt")
    def test_to_ply_missing_directory(self, mock_savetxt, mock_makedirs):
        # Valid data, but we simulate a permission error or IO issue
        output_dir = "output_dir"
        point_cloud = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])

        sfm_instance = Sfm()

        # Mock the sub_dir part
        with patch.object(sfm_instance, "img_loader", new=MagicMock()):
            sfm_instance.img_loader.image_list = ["path/to/image"]

            # Simulate a failure when creating the directory
            mock_makedirs.side_effect = OSError("Directory creation failed")

            # Call the method and check for RuntimeError
            with self.assertRaises(RuntimeError):
                sfm_instance.to_ply(output_dir, point_cloud, colors)

    @patch("os.makedirs")
    @patch("numpy.savetxt")
    def test_to_ply_invalid_color_data(self, mock_savetxt, mock_makedirs):
        # Invalid color data (incorrect shape)
        output_dir = "output_dir"
        point_cloud = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        colors = np.array([[255, 0], [0, 255], [0, 0]])  # Invalid color shape

        sfm_instance = Sfm()

        # Mock the sub_dir part
        with patch.object(sfm_instance, "img_loader", new=MagicMock()):
            sfm_instance.img_loader.image_list = ["path/to/image"]

            # Call the method and expect it to raise an error
            with self.assertRaises(ValueError):
                sfm_instance.to_ply(output_dir, point_cloud, colors)


if __name__ == "__main__":
    unittest.main()
