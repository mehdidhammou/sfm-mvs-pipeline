Collecting workspace information

# README

This project performs Structure from Motion (SfM) and Iterative Closest Point (ICP) alignment on a series of images to generate and merge point clouds.

## Requirements

Ensure you have the required dependencies installed. You can install them using:

```sh
pip install -r requirements.txt
```

## Steps to Run the Project

### 1. Load the Datasets

Ensure your datasets are organized in subdirectories within a main directory. Each subdirectory should contain images for that dataset.

### 2. Calibrate the Camera

Load the chessboard images and calibrate the camera to generate the intrinsic matrix. This step is required for SfM.

```sh
python src/calibrate.py <dataset_path> <chessboard_dims> -s <scale_factor> -o <output_dir>
```

- `<dataset_path>`: Path to the dataset containing chessboard images.
- `<chessboard_dims>`: Dimensions of the chessboard in WxH format (e.g., 4x4).
- `<scale_factor>`: Scale factor for image pyramid downscaling (optional).
- `<output_dir>`: Directory to save camera parameters (optional).

Example:

```sh
python src/calibrate.py chessboard/ 4x4 -s 2 -o camera_params
```

Once the camera is calibrated, copy the instrinsic matrix `K.npy` to each dataset subdirectory.

### 3. Run Structure from Motion (SfM)

Run SfM on the datasets to generate point clouds. Use the following command:

```sh
python src/sfm.py <datasets_dir> <output_dir> -s <downscale_factor>
```

- `<datasets_dir>`: Directory containing subdirectories with datasets.
- `<output_dir>`: Directory to save the output point clouds.
- `<downscale_factor>`: Downscale factor for image loading (optional).

Example:

```sh
python src/sfm.py datasets output -s 2
```

This will generate point clouds for each dataset in the `output` directory. The point clouds will be saved as PLY files in the format `dataset_name.ply`.

### 4. Run ICP to Merge Point Clouds

Merge the generated point clouds using ICP. Use the following command:

```sh
python src/icp.py <input_dir> <output_dir> -i <max_iter> -t <tol> -s <std_ratio> -n <nb_neighbors> -d <threshold>
```

- `<input_dir>`: Path to the directory containing point cloud PLY files.
- `<output_dir>`: Path to the directory to save the merged point cloud.
- `<max_iter>`: Maximum number of ICP iterations (optional).
- `<tol>`: Tolerance for ICP convergence (optional).
- `<std_ratio>`: Standard deviation ratio for statistical outlier removal (optional).
- `<nb_neighbors>`: Number of neighbors to analyze for each point in statistical outlier removal (optional).
- `<threshold>`: Distance threshold for ICP (optional).

Example:

```sh
python src/icp.py point_clouds output
```

### 5. Run Unit Tests

To run the unit tests, use the following command:

```sh
python -m unittest discover tests
```

This will execute all the unit tests in the `tests` directory.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
