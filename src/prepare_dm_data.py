import wandb
import sys
import matplotlib.pyplot as plt
import scprep
import pandas as pd
sys.path.append('../src/')
from evaluate import get_results
from omegaconf import OmegaConf
from main import load_data
from model import AEDist
import numpy as np
import os
import glob
import torch

def prepare_dm_data(cfg, save_path='../dm_data/', folder_path="/wandb/"):
    folder_list = glob.glob(f"{folder_path}*{run.id}*")
    ckpt_files = glob.glob(f"{folder_list[0]}/files/*.ckpt")
    ckpt_path = ckpt_files[0]
    data_path = os.path.join(cfg.data.root, cfg.data.name + cfg.data.filetype)
    data = np.load(data_path, allow_pickle=True)
    # model = AEDist(dim=50, emb_dim=10)
    # model.load_from_checkpoint(ckpt_path)
    model = AEDist.load_from_checkpoint(ckpt_path)
    model.eval()
    x_all = torch.tensor(data['data'], dtype=torch.float32)
    x_pred, z_pred = model(x_all)
    x_pred = x_pred.detach().cpu().numpy()
    z_pred = z_pred.detach().cpu().numpy()
    save_name = f'{save_path}/{cfg.data.name}_{cfg.model.emb_dim}_dm.npz'
    np.savez(save_name, data=z_pred, train_mask=data['is_train'])

if __name__ == "__main__":
    wandb.login()
    api = wandb.Api()

    # Specify your entity, project, and sweep ID
    entity = "xingzhis"
    project = "dmae"
    sweep_id = '9qr8zqxg'

    # Fetch the sweep
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    run_ids = []
    for run in run_ids:
        cfg = OmegaConf.create(run.config)
        