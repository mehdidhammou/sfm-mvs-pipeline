import os
import numpy as np
import cv2


class ImageLoader:
    def __init__(self, img_dir: str, downscale_factor: float):
        # Load camera intrinsic parameters
        with open(os.path.join(img_dir, "K.txt")) as f:
            self.K = np.array(
                [list(map(float, line.split())) for line in f if line.strip()]
            )

        # Load images
        self.image_list = [
            os.path.join(img_dir, image)
            for image in sorted(os.listdir(img_dir))
            if image.lower().endswith((".jpg", ".png"))
        ]

        self.factor = downscale_factor
        self.downscale()

    def downscale(self) -> None:
        """Downscale the intrinsic parameters."""
        self.K[:2, :3] /= self.factor

    def downscale_image(self, image):
        """Downscale the image using pyrDown."""
        for _ in range(int(np.log2(self.factor))):
            image = cv2.pyrDown(image)
        return image