import unittest
import os
import numpy as np
import cv2
from tempfile import TemporaryDirectory
from src.utils import ImageLoader


class TestImageLoader(unittest.TestCase):
    def setUp(self):
        self.test_dir = TemporaryDirectory()
        self.img_dir = self.test_dir.name
        self.K_path = os.path.join(self.img_dir, "K.npy")

        # Create mock intrinsic parameter file using numpy
        K = np.array([[1000, 0, 512], [0, 1000, 384], [0, 0, 1]], dtype=float)
        np.save(self.K_path, K)

        # Create mock images
        self.mock_images = []
        for i in range(3):
            img_path = os.path.join(self.img_dir, f"image_{i}.jpg")
            image = np.random.randint(0, 256, (1024, 768, 3), dtype=np.uint8)
            cv2.imwrite(img_path, image)
            self.mock_images.append(img_path)

        self.downscale_factor = 2.0

    def tearDown(self):
        self.test_dir.cleanup()

    def test_intrinsic_parameters_loading(self):
        loader = ImageLoader(self.img_dir, self.K_path, self.downscale_factor)
        expected_K = np.array([[1000, 0, 512], [0, 1000, 384], [0, 0, 1]], dtype=float)
        loader.K[:2, :3] *= self.downscale_factor
        np.testing.assert_array_equal(loader.K, expected_K)

    def test_image_loading(self):
        loader = ImageLoader(self.img_dir, self.K_path, self.downscale_factor)
        self.assertEqual(len(loader.image_list), len(self.mock_images))
        for loaded_image, mock_image in zip(loader.image_list, self.mock_images):
            self.assertTrue(loaded_image.endswith(mock_image.split(os.sep)[-1]))

    def test_downscale_intrinsics(self):
        loader = ImageLoader(self.img_dir, self.K_path, self.downscale_factor)
        expected_K = np.array([[500, 0, 256], [0, 500, 192], [0, 0, 1]])
        np.testing.assert_array_almost_equal(loader.K, expected_K)

    def test_downscale_image(self):
        loader = ImageLoader(self.img_dir, self.K_path, self.downscale_factor)
        original_image = cv2.imread(self.mock_images[0])
        downscaled_image = loader.downscale_image(original_image)
        expected_shape = (
            original_image.shape[0] // 2,
            original_image.shape[1] // 2,
            original_image.shape[2],
        )
        self.assertEqual(downscaled_image.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
