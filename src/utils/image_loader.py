import os
import numpy as np
import cv2


class ImageLoader:
    def __init__(self, img_dir: str, K_path: str, downscale_factor: float = 2.0):
        self.img_dir = img_dir
        self.factor = downscale_factor

        self.K: np.ndarray = np.load(K_path)
        self.image_list = self.load_images()
        self.downscale()

    def load_images(self) -> list[str]:
        """Load the images from the directory."""
        if not os.path.isdir(self.img_dir):
            raise NotADirectoryError(f"The directory {self.img_dir} does not exist.")

        return [
            os.path.join(self.img_dir, image)
            for image in sorted(os.listdir(self.img_dir))
            if image.lower().endswith((".jpg", ".png"))
        ]

    def downscale(self) -> None:
        """Downscale the intrinsic parameters based on the factor."""
        self.K[:2, :3] /= self.factor

    def downscale_image(self, image) -> np.ndarray:
        """Downscale the image using pyrDown."""
        num_scales = int(np.log2(self.factor))
        for _ in range(num_scales):
            image = cv2.pyrDown(image)
        return image
