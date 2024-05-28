import numpy as np
import torch
from autometric.geodesics import DjikstraGeodesic
import pathlib
import pickle
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cfg_main_data_name', type=str, default='true_4_paths_17580_3000_1_all')
parser.add_argument('--result_path', type=str, default='results_djikstra')
parser.add_argument('--data_root', type=str, default='../../synthetic_data4/')
parser.add_argument('--n_steps', type=int, default=20)

args = parser.parse_args()
cfg_main_data_name = args.cfg_main_data_name
result_path = args.result_path
data_root = args.data_root
n_steps = args.n_steps

n_total = 3000
n_geods = 20
np.random.seed(3234)
start_idx = np.random.choice(n_total, n_geods)
end_idx = np.random.choice(n_total, n_geods)

res_path = f'{result_path}/{cfg_main_data_name}/'
pathlib.Path(res_path).mkdir(parents=True, exist_ok=True)

data = np.load(f"{data_root}/{cfg_main_data_name}.npz")
x = torch.tensor(data['data'])
start_points = x[start_idx,:]
end_points = x[end_idx,:]

DG = DjikstraGeodesic(x)
gs, ls = DG.geodesics(start_points, end_points, np.linspace(0,1,n_steps))

np.savez(f"{res_path}/{dijkstra}.npz", geodesic_lengths=ls.cpu().numpy(), geodesics=gs.cpu().numpy())
