import argparse
import os
from glob import glob

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from tomlkit import boolean
from tqdm import tqdm

from utils.image_loader import ImageLoader


class Sfm:
    def __init__(self, img_loader: ImageLoader, output_dir: str) -> None:
        self.img_loader = img_loader
        self.output_dir = output_dir

    def triangulation(
        self, point_2d_1, point_2d_2, projection_matrix_1, projection_matrix_2
    ) -> tuple:
        """
        Triangulates 3d points from 2d vectors and projection matrices
        returns projection matrix of first camera, projection matrix of second camera, and the point cloud
        """
        pt_cloud = cv2.triangulatePoints(
            point_2d_1, point_2d_2, projection_matrix_1.T, projection_matrix_2.T
        )
        return projection_matrix_1.T, projection_matrix_2.T, (pt_cloud / pt_cloud[3])

    def PnP(self, obj_point, image_point, K, dist_coeff, rot_vector, initial) -> tuple:
        """
        Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
        returns rotational matrix, translational matrix, image points, object points, rotational vector
        """
        if initial == 1:
            obj_point = obj_point[:, 0, :]
            image_point = image_point.T
            rot_vector = rot_vector.T
        _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(
            obj_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE
        )
        # Converts a rotation matrix to a rotation vector or vice versa
        rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)

        if inlier is not None:
            image_point = image_point[inlier[:, 0]]
            obj_point = obj_point[inlier[:, 0]]
            rot_vector = rot_vector[inlier[:, 0]]
        return rot_matrix, tran_vector, image_point, obj_point, rot_vector

    def reprojection_error(
        self, obj_points, image_points, transform_matrix, K, homogenity
    ) -> tuple:
        """
        Calculates the reprojection error ie the distance between the projected points and the actual points.
        returns total error, object points
        """
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        if homogenity == 1:
            obj_points = cv2.convertPointsFromHomogeneous(obj_points.T)
        image_points_calc, _ = cv2.projectPoints(
            obj_points, rot_vector, tran_vector, K, None
        )
        image_points_calc = np.float32(image_points_calc[:, 0, :])
        total_error = cv2.norm(
            image_points_calc,
            np.float32(image_points.T) if homogenity == 1 else np.float32(image_points),
            cv2.NORM_L2,
        )
        return total_error / len(image_points_calc), obj_points

    def optimal_reprojection_error(self, obj_points) -> np.array:
        """
        calculates of the reprojection error during bundle adjustment
        returns error
        """
        transform_matrix = obj_points[0:12].reshape((3, 4))
        K = obj_points[12:21].reshape((3, 3))
        rest = int(len(obj_points[21:]) * 0.4)
        p = obj_points[21 : 21 + rest].reshape((2, int(rest / 2))).T
        obj_points = obj_points[21 + rest :].reshape(
            (int(len(obj_points[21 + rest :]) / 3), 3)
        )
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        image_points, _ = cv2.projectPoints(
            obj_points, rot_vector, tran_vector, K, None
        )
        image_points = image_points[:, 0, :]
        error = [(p[idx] - image_points[idx]) ** 2 for idx in range(len(p))]
        return np.array(error).ravel() / len(p)

    def bundle_adjustment(
        self, _3d_point, opt, transform_matrix_new, K, r_error
    ) -> tuple:
        """
        Bundle adjustment for the image and object points
        returns object points, image points, transformation matrix
        """
        opt_variables = np.hstack((transform_matrix_new.ravel(), K.ravel()))
        opt_variables = np.hstack((opt_variables, opt.ravel()))
        opt_variables = np.hstack((opt_variables, _3d_point.ravel()))

        values_corrected = least_squares(
            self.optimal_reprojection_error, opt_variables, gtol=r_error
        ).x
        K = values_corrected[12:21].reshape((3, 3))
        rest = int(len(values_corrected[21:]) * 0.4)
        return (
            values_corrected[21 + rest :].reshape(
                (int(len(values_corrected[21 + rest :]) / 3), 3)
            ),
            values_corrected[21 : 21 + rest].reshape((2, int(rest / 2))).T,
            values_corrected[0:12].reshape((3, 4)),
        )

    def to_ply(self, point_cloud: np.ndarray, colors: np.ndarray) -> None:
        """
        Generates a .ply file to visualize the point cloud using Open3D.

        Args:
            point_cloud (np.ndarray): 3D point cloud data (Nx3).
            colors (np.ndarray): Corresponding RGB color data (Nx3).
        """
        try:
            # Scale the point cloud data
            scaled_points = point_cloud.reshape(-1, 3) * 200

            # Reshape color data
            reshaped_colors = colors.reshape(-1, 3)

            # Create Open3D point cloud object
            pcd = o3d.geometry.PointCloud()

            # Set the points and colors for the point cloud
            pcd.points = o3d.utility.Vector3dVector(scaled_points)
            pcd.colors = o3d.utility.Vector3dVector(
                reshaped_colors / 255.0
            )  # Normalize colors to [0, 1]

            # Filter out points based on distance from the mean
            mean_position = np.mean(scaled_points, axis=0)
            distances = np.linalg.norm(scaled_points - mean_position, axis=1)
            threshold = np.mean(distances) + 300
            pcd = pcd.select_by_index(np.where(distances < threshold)[0])

            # Generate output file path
            os.makedirs(self.output_dir, exist_ok=True)
            sub_dir = os.path.basename(self.img_loader.img_dir)
            file_path = os.path.join(self.output_dir, f"{sub_dir}.ply")

            # Write point cloud to .ply file
            o3d.io.write_point_cloud(file_path, pcd)
            print(f"Successfully generated .ply file at: {file_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to generate .ply file: {e}")

    def common_points(self, image_points_1, image_points_2, image_points_3) -> tuple:
        """
        Finds the common points between image 1 and 2 , image 2 and 3
        returns common points of image 1-2, common points of image 2-3, mask of common points 1-2 , mask for common points 2-3
        """
        cm_points_1 = []
        cm_points_2 = []
        for i in range(image_points_1.shape[0]):
            a = np.where(image_points_2 == image_points_1[i, :])
            if a[0].size != 0:
                cm_points_1.append(i)
                cm_points_2.append(a[0][0])

        mask_array_1 = np.ma.array(image_points_2, mask=False)
        mask_array_1.mask[cm_points_2] = True
        mask_array_1 = mask_array_1.compressed()
        mask_array_1 = mask_array_1.reshape(int(mask_array_1.shape[0] / 2), 2)

        mask_array_2 = np.ma.array(image_points_3, mask=False)
        mask_array_2.mask[cm_points_2] = True
        mask_array_2 = mask_array_2.compressed()
        mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)
        print(" Shape New Array", mask_array_1.shape, mask_array_2.shape)
        return np.array(cm_points_1), np.array(cm_points_2), mask_array_1, mask_array_2

    def find_features(self, image_0, image_1) -> tuple:
        """
        Feature detection using the sift algorithm and KNN
        return keypoints(features) of image1 and image2
        """

        sift = cv2.SIFT.create()
        key_points_0, desc_0 = sift.detectAndCompute(
            cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None
        )
        key_points_1, desc_1 = sift.detectAndCompute(
            cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None
        )

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_0, desc_1, k=2)
        feature = []
        for m, n in matches:
            if m.distance < 0.70 * n.distance:
                feature.append(m)

        return np.float32([key_points_0[m.queryIdx].pt for m in feature]), np.float32(
            [key_points_1[m.trainIdx].pt for m in feature]
        )

    def run_sfm(self, enable_bundle_adjustment: boolean = False):
        # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        pose_array = self.img_loader.K.ravel()
        transform_matrix_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        transform_matrix_1 = np.empty((3, 4))

        pose_0 = np.matmul(self.img_loader.K, transform_matrix_0)
        pose_1 = np.empty((3, 4))
        total_points = np.zeros((1, 3))
        total_colors = np.zeros((1, 3))

        image_0 = self.img_loader.downscale_image(
            cv2.imread(self.img_loader.image_list[0])
        )
        image_1 = self.img_loader.downscale_image(
            cv2.imread(self.img_loader.image_list[1])
        )

        feature_0, feature_1 = self.find_features(image_0, image_1)

        # Essential matrix
        essential_matrix, em_mask = cv2.findEssentialMat(
            feature_0,
            feature_1,
            self.img_loader.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.4,
            mask=None,
        )
        feature_0 = feature_0[em_mask.ravel() == 1]
        feature_1 = feature_1[em_mask.ravel() == 1]

        _, rot_matrix, tran_matrix, em_mask = cv2.recoverPose(
            essential_matrix, feature_0, feature_1, self.img_loader.K
        )
        feature_0 = feature_0[em_mask.ravel() > 0]
        feature_1 = feature_1[em_mask.ravel() > 0]
        transform_matrix_1[:3, :3] = np.matmul(rot_matrix, transform_matrix_0[:3, :3])
        transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + np.matmul(
            transform_matrix_0[:3, :3], tran_matrix.ravel()
        )

        pose_1 = np.matmul(self.img_loader.K, transform_matrix_1)

        feature_0, feature_1, points_3d = self.triangulation(
            pose_0, pose_1, feature_0, feature_1
        )
        error, points_3d = self.reprojection_error(
            points_3d, feature_1, transform_matrix_1, self.img_loader.K, homogenity=1
        )
        # ideally error < 1
        print("REPROJECTION ERROR: ", error)
        _, _, feature_1, points_3d, _ = self.PnP(
            points_3d,
            feature_1,
            self.img_loader.K,
            np.zeros((5, 1), dtype=np.float32),
            feature_0,
            initial=1,
        )

        total_images = len(self.img_loader.image_list) - 2
        pose_array = np.hstack(
            (np.hstack((pose_array, pose_0.ravel())), pose_1.ravel())
        )

        threshold = 0.5
        for i in tqdm(range(total_images)):
            image_2 = self.img_loader.downscale_image(
                cv2.imread(self.img_loader.image_list[i + 2])
            )
            features_cur, features_2 = self.find_features(image_1, image_2)

            if i != 0:
                feature_0, feature_1, points_3d = self.triangulation(
                    pose_0, pose_1, feature_0, feature_1
                )
                feature_1 = feature_1.T
                points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
                points_3d = points_3d[:, 0, :]

            cm_points_0, cm_points_1, cm_mask_0, cm_mask_1 = self.common_points(
                feature_1, features_cur, features_2
            )

            print("Common Points: ", cm_points_0.shape, cm_points_1.shape)

            cm_points_2 = features_2[cm_points_1]
            cm_points_cur = features_cur[cm_points_1]

            rot_matrix, tran_matrix, cm_points_2, points_3d, cm_points_cur = self.PnP(
                points_3d[cm_points_0],
                cm_points_2,
                self.img_loader.K,
                np.zeros((5, 1), dtype=np.float32),
                cm_points_cur,
                initial=0,
            )
            transform_matrix_1 = np.hstack((rot_matrix, tran_matrix))
            pose_2 = np.matmul(self.img_loader.K, transform_matrix_1)

            error, points_3d = self.reprojection_error(
                points_3d,
                cm_points_2,
                transform_matrix_1,
                self.img_loader.K,
                homogenity=0,
            )

            cm_mask_0, cm_mask_1, points_3d = self.triangulation(
                pose_1, pose_2, cm_mask_0, cm_mask_1
            )
            error, points_3d = self.reprojection_error(
                points_3d,
                cm_mask_1,
                transform_matrix_1,
                self.img_loader.K,
                homogenity=1,
            )
            print("Reprojection Error: ", error)
            pose_array = np.hstack((pose_array, pose_2.ravel()))
            # takes a long time to run
            if enable_bundle_adjustment:
                points_3d, cm_mask_1, transform_matrix_1 = self.bundle_adjustment(
                    points_3d,
                    cm_mask_1,
                    transform_matrix_1,
                    self.img_loader.K,
                    threshold,
                )
                pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)
                error, points_3d = self.reprojection_error(
                    points_3d,
                    cm_mask_1,
                    transform_matrix_1,
                    self.img_loader.K,
                    homogenity=0,
                )
                print("Bundle Adjusted error: ", error)
                total_points = np.vstack((total_points, points_3d))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left])
                total_colors = np.vstack((total_colors, color_vector))
            else:
                total_points = np.vstack((total_points, points_3d[:, 0, :]))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left.T])
                total_colors = np.vstack((total_colors, color_vector))

            transform_matrix_0 = np.copy(transform_matrix_1)
            pose_0 = np.copy(pose_1)

            image_0 = np.copy(image_1)
            image_1 = np.copy(image_2)
            feature_0 = np.copy(features_cur)
            feature_1 = np.copy(features_2)
            pose_1 = np.copy(pose_2)
            folder_name = os.path.basename(self.img_loader.img_dir)
            # cv2.imshow(folder_name, image_2)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

        print(total_points.shape, total_colors.shape)
        self.to_ply(total_points, total_colors)


def get_argparser():
    """Sets up the argument parser."""
    parser = argparse.ArgumentParser(
        description="Run Structure from Motion (SfM) on datasets."
    )
    parser.add_argument(
        "datasets_dir",
        type=str,
        help="Directory containing subdirectories with datasets.",
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory to save the output point clouds."
    )
    parser.add_argument(
        "-s",
        "--downscale_factor",
        type=int,
        default=2,
        help="Downscale factor for image loading (default is 1.0, no downscaling).",
    )
    return parser


def check_k_matrix_exists(k_path):
    """Check if K.npy exists."""
    if not os.path.exists(k_path):
        raise FileNotFoundError(f"Calibration matrix 'K.npy' not found at: {k_path}")


if __name__ == "__main__":
    args = get_argparser().parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Get datasets from the specified 'datasets_dir' folder
    datasets = glob(os.path.join(args.datasets_dir, "*"))

    for idx, dataset in enumerate(datasets):
        K_path = os.path.join(dataset, "K.npy")
        check_k_matrix_exists(K_path)
        print(f"Dataset {idx + 1}, path: {dataset}, K matrix: {K_path}")

        # Load the ImageLoader with the downscale factor (optional)
        img_loader = ImageLoader(
            dataset, K_path, downscale_factor=args.downscale_factor
        )

        # Initialize and run the SfM process
        sfm = Sfm(img_loader, args.output_dir)
        sfm.run_sfm()
