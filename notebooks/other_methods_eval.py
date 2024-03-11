import wandb
import sys
import matplotlib.pyplot as plt
import scprep
import pandas as pd
sys.path.append('../src/')
from evaluate import get_results
from omegaconf import OmegaConf
import numpy as np
import os
import glob
import demap
from tqdm import tqdm
from evaluation import compute_all_metrics, get_noiseless_name, get_ambient_name
import torch
from model import AEProb, Decoder

import phate
from sklearn.manifold import TSNE
import umap
from other_methods import DiffusionMap

# for other embedding methods
class Model(): 
    def __init__(self, method, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.method = method
        assert method in ['phate', 'umap', 'tsne', 'diffusion_map']

        if method == 'phate':
            self.model = phate.PHATE(**kwargs)
        elif method == 'umap':
            self.model = umap.UMAP(**kwargs)
        elif method == 'tsne':
            self.model = TSNE(**kwargs)
        elif method == 'diffusion_map':
            self.model = DiffusionMap(**kwargs)

    def encode(self, x):
        return self.model.fit_transform(x)

root_path = '../affinity_matching_results_xingzhi/results/'
data_paths = os.listdir(root_path)

results = []
for data_path1 in tqdm(data_paths):
    if data_path1.startswith('sepa_'):
        data_name = data_path1[14:-13]
        data_root = '../synthetic_data2/'
        data_path = os.path.join(data_root, data_name + '.npz')
        noiseless_path = os.path.join(data_root, get_noiseless_name(data_name) + '.npz')
        ambient_path = os.path.join(data_root, get_ambient_name(data_name) + '.npy')
        
        model = Model('phate')
        res_dict = compute_all_metrics(model, data_path, noiseless_path, ambient_path)
        results.append(res_dict)

res_df = pd.DataFrame(results)
res_df.to_csv("affinity_synth_results.csv", index=False)

res_df = res_df.sort_values(['dataset', 'bcv', 'dropout', 'probmethod'])
rounded_res_df = res_df.select_dtypes(include=['float64']).round(3)
for col in res_df.select_dtypes(exclude=['float64']).columns:
    rounded_res_df[col] = res_df[col]

rounded_res_df = rounded_res_df[res_df.columns]
rounded_res_df.to_csv("affinity_synth_results_rounded.csv", index=False)
