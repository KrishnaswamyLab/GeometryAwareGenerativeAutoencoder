import numpy as np
import pathlib
import argparse
import sys
from tqdm import tqdm
from itertools import product
sys.path.append('../../src')
from negative_sampling import add_negative_samples, make_hi_freq_noise
from diffusionmap import DiffusionMap
import phate
from plotly3d.plot import scatter

def process_data(data_path, noise_type, noise_level, subset_rate, seed, mask_dists):
    data0 = np.load(data_path)
    data_dict = {f: data0[f] for f in data0.files}
    data_dict = add_negative_samples(data_dict, subset_rate=subset_rate, noise_rate=noise_level, seed=seed, noise=noise_type, mask_dists=mask_dists)
    savefldr = f'../../data/negative_sampling_new/dist_mask_{mask_dists}/{noise_type}/{noise_level}/'
    pathlib.Path(savefldr).mkdir(parents=True, exist_ok=True)
    data_name = data_path.split('/')[-1].split('.')[0]
    np.savez(f"{savefldr}/{data_name}.npz", **data_dict)

def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--data_path', type=str, help='Path to the data file')
    parser.add_argument('--noise_type', type=str, help='Type of noise')
    parser.add_argument('--noise_level', type=float, help='Level of noise')
    parser.add_argument('--subset_rate', type=float, default=0.5, help='Rate of subset')
    parser.add_argument('--seed', type=int, default=42, help='Seed for randomness')
    parser.add_argument('--mask_dists', action='store_true', help='Mask distances')
    args = parser.parse_args()
    process_data(args.data_path, args.noise_type, args.noise_level, args.subset_rate, args.seed, args.mask_dists)

if __name__ == "__main__":
    main()
