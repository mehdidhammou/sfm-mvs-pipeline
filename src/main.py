import argparse
from icp import run_icp


def get_argparser() -> argparse.Namespace:
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


if __name__ == "__main__":
    run_icp(vars(get_argparser()))
