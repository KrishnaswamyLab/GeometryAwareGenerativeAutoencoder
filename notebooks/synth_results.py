import wandb
import sys
import matplotlib.pyplot as plt
import scprep
import pandas as pd
sys.path.append('../src/')
from evaluate import get_results
from omegaconf import OmegaConf
from main import load_data, make_model
import numpy as np
import os
import glob
import demap
from tqdm import tqdm
# from evaluation import compute_encoding_metrics, get_dataset_contents, get_noiseless_name, get_ambient_name, get_data_config, eval_results, compute_recon_metric
from evaluation import compute_all_metrics, get_noiseless_name, get_ambient_name
from transformations import NonTransform

# Initialize wandb (replace 'your_entity' and 'your_project' with your specific details)
wandb.login()
api = wandb.Api()

# Specify your entity, project, and sweep ID
entity = "xingzhis"
project = "dmae"
sweep_id = 'wgysuau8'

# Fetch the sweep
sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

run_ids = [run.id for run in sweep.runs]

results = []

for i in tqdm(range(len(sweep.runs))):
    run = sweep.runs[i]
    cfg = OmegaConf.create(run.config)
    folder_path = "../src/wandb/"
    try:
        folder_list = glob.glob(f"{folder_path}*{run.id}*")
        ckpt_files = glob.glob(f"{folder_list[0]}/files/*.ckpt")
        ckpt_path = ckpt_files[0]
    except:
        print(f"No checkpoint found for run {run.id}")
    cfg = OmegaConf.create(run.config)
    data_root = '../synthetic_data2/'
    data_path = os.path.join(data_root, cfg.data.name + cfg.data.filetype)
    noiseless_path = os.path.join(data_root, get_noiseless_name(cfg.data.name) + cfg.data.filetype)
    ambient_path = os.path.join(data_root, get_ambient_name(cfg.data.name) + '.npy')
    pp = NonTransform()
    emb_dim = cfg.model.emb_dim
    dist_std = 1.
    input_dim = 100
    model = make_model(cfg, input_dim, emb_dim, pp, dist_std, from_checkpoint=True, checkpoint_path=ckpt_path)
    res_dict = compute_all_metrics(model, data_path, noiseless_path, ambient_path)
    results.append(res_dict)

res_df = pd.DataFrame(results)
res_df.to_csv("synth_results_w_recon.csv", index=False)

res_df = res_df.sort_values(['dataset', 'bcv', 'dropout', 'reconstr_weight'])
# Round all numeric columns to 3 decimals, excluding strings
rounded_res_df = res_df.select_dtypes(include=['float64']).round(3)
# Re-attach the non-numeric columns to the rounded DataFrame
for col in res_df.select_dtypes(exclude=['float64']).columns:
    rounded_res_df[col] = res_df[col]

# Reorder columns to match original DataFrame
rounded_res_df = rounded_res_df[res_df.columns]
rounded_res_df.to_csv("synth_results_w_recon_rounded.csv", index=False)