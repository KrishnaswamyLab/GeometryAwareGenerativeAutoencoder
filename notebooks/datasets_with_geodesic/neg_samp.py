import numpy as np
import sys
import os
sys.path.append('../../src/')
from data_convert import convert_data
from negative_sampling import add_negative_samples, make_hi_freq_noise
import pathlib

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, required=True)
args = parser.parse_args()
data_name = args.data_name

seed = 32

ae_dir = '/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/data/neurips_results/toy/ae/'
disc_dir = '/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/data/neurips_results/toy/disc/'
pathlib.Path(ae_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(disc_dir).mkdir(parents=True, exist_ok=True)
# folder = '../../data/gt_geodesic/'
folder = '/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/data/neurips_results/toy/'

data = np.load(f'{folder}/{data_name}', allow_pickle=True)
data_dict_raw = {f:data[f] for f in data.files}

points = data_dict_raw['X']
data_dict0 = convert_data(points)

for key, value in data_dict_raw.items():
    if key == 'X':
        continue
    data_dict0[key] = value

# noise_type = 'hi-freq-no-add'
noise_type = 'gaussian'
noise_level = 0.5
mask_dist = False
data_dict = add_negative_samples(data_dict0.copy(), subset_rate=1., noise_rate=noise_level, seed=seed, noise=noise_type, mask_dists=mask_dist, shell=True)

np.savez(f"{ae_dir}/{data_name}", **data_dict0)
np.savez(f"{disc_dir}/{data_name}", **data_dict)