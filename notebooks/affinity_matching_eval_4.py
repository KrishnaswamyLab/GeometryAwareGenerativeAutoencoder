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
import re

class Model():
    def __init__(self, encoder, decoder):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
    def encode(self, x):
        return self.encoder.encode(x)
    def decode(self, x):
        return self.decoder(x)
    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

root_path = '/gpfs/gibbs/pi/krishnaswamy_smita/dl2282/dmae/results/'
data_paths_all = os.listdir(root_path)
data_paths = [dp for dp in data_paths_all if dp.startswith('sepa_gaussian_jsd_a1.0_knn5_noisy_')]

results = []
for data_path1 in tqdm(data_paths):
    if data_path1.startswith('sepa_'):
        enc_path = os.path.join(root_path, data_path1, 'model.ckpt')
        dec_path = os.path.join(root_path, data_path1, 'decoder.ckpt')
        encoder_dict = torch.load(enc_path)
        decoder_dict = torch.load(dec_path)
        
        # Regex pattern to extract the values
        pattern = r"sepa_(?P<prob_method>\w+)_a(?P<alpha>[\d.]+)_knn(?P<knn>\d+)_(?P<noisy_path>.+)"

        # Perform regex search
        match = re.search(pattern, data_path1)

        if match:
            # Extracting the values
            prob_method = match.group("prob_method")
            alpha = match.group("alpha")
            knn = match.group("knn")
            noisy_path = match.group("noisy_path")

        data_name = noisy_path[:-4]
        probmtd = prob_method
        
        data_root = '../synthetic_data3/'
        data_path = os.path.join(data_root, data_name + '.npz')
        noiseless_path = os.path.join(data_root, get_noiseless_name(data_name) + '.npz')
        ambient_path = os.path.join(data_root, get_ambient_name(data_name) + '.npy')
        encoder = AEProb(dim=100, emb_dim=2, layer_widths=[256, 128, 64], activation_fn=torch.nn.ReLU(), prob_method=probmtd, dist_reconstr_weights=[1.0,0.0,0.], )
        encoder.load_state_dict(encoder_dict)
        decoder = Decoder(dim=100, emb_dim=2, layer_widths=[256, 128, 64][::-1], activation_fn=torch.nn.ReLU())
        decoder.load_state_dict(decoder_dict)
        model = Model(encoder, decoder)
        res_dict = compute_all_metrics(model, data_path, noiseless_path, ambient_path)
        res_dict['probmethod'] = probmtd
        res_dict['alpha'] = alpha
        res_dict['knn'] = knn
        
        results.append(res_dict)

res_df = pd.DataFrame(results)
res_df.to_csv("affinity_synth_results_4.csv", index=False)

res_df = res_df.sort_values(['seedmethod', 'bcv', 'dropout', 'probmethod'])
rounded_res_df = res_df.select_dtypes(include=['float64']).round(3)
for col in res_df.select_dtypes(exclude=['float64']).columns:
    rounded_res_df[col] = res_df[col]

rounded_res_df = rounded_res_df[res_df.columns]
rounded_res_df.to_csv("affinity_synth_results_4_rounded.csv", index=False)
