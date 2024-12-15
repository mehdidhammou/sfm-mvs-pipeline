import argparse
import glob
import os
import warnings

import cv2
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")


def calibrate_camera(dataset_path, chessboard_dims, scale_factor, output_dir):
    # Prepare dataset path and fetch images
    dataset_path = os.path.join(dataset_path, "*.jpg")
    images = glob.glob(dataset_path)

    assert len(images) > 0, f"No images found in {dataset_path}"
    print(f"Found {len(images)} images in {dataset_path}")

    CHECKERBOARD = tuple(map(int, chessboard_dims.split("x")))

    # Create vectors to store 3D and 2D points for each checkerboard image
    objpoints = []
    imgpoints = []

    # Define world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)

    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for image in tqdm(images, desc="Processing images"):
        img = cv2.imread(image)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Scale down the image based on the scale factor
        for _ in range(scale_factor):
            gray = cv2.pyrDown(gray)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    if not objpoints or not imgpoints:
        print("No valid chessboard corners were found.")
        return

    # Camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # Save camera calibration data
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "K.npy"), mtx)
    np.save(os.path.join(output_dir, "dist.npy"), dist)
    np.save(os.path.join(output_dir, "rvecs.npy"), rvecs)
    np.save(os.path.join(output_dir, "tvecs.npy"), tvecs)

    # Compute total reprojection error
    error = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error.append(cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2))

    print("Total error:", np.mean(error))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration script.")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset containing chessboard images.",
    )
    parser.add_argument(
        "chessboard_dims",
        type=str,
        help="Chessboard dimensions in WxH format (e.g., 4x4).",
    )
    parser.add_argument(
        "-s",
        "--scale_factor",
        type=int,
        required=False,
        default=1,
        help="Scale factor for image pyramid downscaling.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        default="camera_params",
        help="Directory to save camera parameters.",
    )

    args = parser.parse_args()
    calibrate_camera(
        args.dataset_path, args.chessboard_dims, args.scale_factor, args.output
    )
