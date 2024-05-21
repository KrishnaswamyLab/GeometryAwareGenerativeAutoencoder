from autometric.datasets import export_dataset
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Export dataset based on specified parameters.')
    
    parser.add_argument('--foldername', type=str, help='Name of the folder to save the dataset.')
    parser.add_argument('--num_geodesics', type=int, default=20, help='Number of geodesics.')
    parser.add_argument('--num_points_per_geodesic', type=int, default=3000, help='Number of points per geodesic.')
    parser.add_argument('--seed', type=int, default=480851, help='Seed for random number generation.')
    parser.add_argument('--get_geod', type=bool, default=True, help='Flag to get geodesics.')
    parser.add_argument('--rot_dim', type=int, default=None, help='Rotation dimension.')
    parser.add_argument('--noise', type=float, default=0, help='Noise level.')
    parser.add_argument('--mfd', type=str, default='Hemisphere', help='Manifold type.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    export_dataset(
        foldername=args.foldername,
        num_geodesics=args.num_geodesics,
        num_points_per_geodesic=args.num_points_per_geodesic,
        seed=args.seed,
        get_geod=args.get_geod,
        rot_dim=args.rot_dim,
        noise=args.noise,
        mfd=args.mfd,
    )
