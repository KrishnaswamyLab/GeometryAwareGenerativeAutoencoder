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

from main import load_data, make_model
from data import dataloader_from_pc
from procrustes import Procrustes

from transformations import LogTransform, NonTransform, StandardScaler, MinMaxScaler, PowerTransformer, KernelTransform

from omegaconf import OmegaConf
import numpy as np
import os
import glob
from scipy.spatial.distance import pdist, squareform

def get_results(run):
    cfg = OmegaConf.create(run.config)
    folder_path = "../src/wandb/"
    try:
        folder_list = glob.glob(f"{folder_path}*{run.id}*")
        ckpt_files = glob.glob(f"{folder_list[0]}/files/*.ckpt")
        ckpt_path = ckpt_files[0]
    except:
        print(f"No checkpoint found for run {run.id}")
        return None, None, None
    allloader, _, X, phate_coords, colors, dist, pp = load_data(cfg, load_all=True)
    emb_dim = phate_coords.shape[1]
    data_path = os.path.join(cfg.data.root, cfg.data.name + cfg.data.filetype)
    data = np.load(data_path, allow_pickle=True)
    dist_std = np.std(data['dist'].flatten())
    model = make_model(cfg, X.shape[1], emb_dim, pp, dist_std, from_checkpoint=True, checkpoint_path=ckpt_path)
    model.eval()
    x_all = next(iter(allloader))['x']
    x_pred, z_pred = model(x_all)
    x_pred = x_pred.detach().cpu().numpy()
    z_pred = z_pred.detach().cpu().numpy()
    data_all = data
    data_path_train = os.path.join(cfg.data.root, cfg.data.name + cfg.data.filetype)
    train_mask = data_all['is_train']
    test_mask = ~data_all['is_train']
    procrustes = Procrustes()
    phate_proc_train, z_hat, disparity = procrustes.fit_transform(data_all['phate'][train_mask], z_pred[train_mask])
    zhat_all = procrustes.transform(z_pred)
    dist_pred = squareform(pdist(zhat_all))
    dist_true = squareform(pdist(data_all['phate']))
    test_test_mask = test_mask[:,None] * test_mask[None,:]
    test_train_mask = test_mask[:,None] * train_mask[None,:]
    train_train_mask = train_mask[:,None] * train_mask[None,:]
    test_all_mask = test_mask[:,None] * np.ones_like(test_mask)
    eps = 1e-10
    dist_mape_test_test = (np.abs(dist_true - dist_pred + eps) / (dist_true + eps) * test_test_mask).sum() / test_test_mask.sum()
    dist_mape_test_train = (np.abs(dist_true - dist_pred + eps) / (dist_true + eps) * test_train_mask).sum() / test_train_mask.sum()
    dist_mape_train_train = (np.abs(dist_true - dist_pred + eps) / (dist_true + eps) * train_train_mask).sum() / train_train_mask.sum()
    dist_mape_test_overall = (np.abs(dist_true - dist_pred + eps) / (dist_true + eps) * test_all_mask).sum() / test_all_mask.sum()
    dist_rmse_test_test = np.sqrt(((dist_true - dist_pred)**2 * test_test_mask).sum()/ test_test_mask.sum())
    dist_rmse_test_train = np.sqrt(((dist_true - dist_pred)**2 * test_train_mask).sum() / test_train_mask.sum())
    dist_rmse_train_train = np.sqrt(((dist_true - dist_pred)**2 * train_train_mask).sum() / train_train_mask.sum())
    test_rmse = np.sqrt((data_all['phate'][test_mask] - zhat_all[test_mask])**2).mean()
    res = dict(
        data=cfg.data.name,
        preprocess=cfg.data.preprocess,
        kernel=cfg.data.kernel.type if cfg.data.preprocess == 'kernel' else None,
        sigma=cfg.data.kernel.sigma if cfg.data.preprocess == 'kernel' else 0,
        dist_recon_weight = cfg.model.dist_reconstr_weights,
        model_type = cfg.model.type,
        dist_mape_test_test=dist_mape_test_test,
        dist_mape_test_train=dist_mape_test_train,
        dist_mape_test_overall=dist_mape_test_overall,
        dist_mape_train_train=dist_mape_train_train,
        dist_rmse_test_test=dist_rmse_test_test,
        dist_rmse_test_train=dist_rmse_test_train,
        dist_rmse_train_train=dist_rmse_train_train,
        test_rmse=test_rmse,
        train_mask=train_mask
    )
    plot_data = dict(
        phate_true = data_all['phate'][test_mask],
        phate_pred = zhat_all[test_mask],
        colors = data_all['colors'][test_mask],
        colors_train = data_all['colors'][train_mask],
        dist_true_test_test = dist_true[test_mask][:,test_mask],
        dist_pred_test_test = dist_pred[test_mask][:,test_mask],
        dist_true_test_train = dist_true[test_mask][:,train_mask],
        dist_pred_test_train = dist_pred[test_mask][:,train_mask],
        phate_true_train = data_all['phate'][train_mask],
        phate_pred_train = zhat_all[train_mask],
        dist_true_train_train = dist_true[train_mask][:,train_mask],
        dist_pred_train_train = dist_pred[train_mask][:,train_mask],
    )
    return res, plot_data, cfg

def rename_string(s):
    # Split the string into parts
    parts = s.split('_')
    
    # Replace "noisy" with "true"
    parts[0] = "true"
    
    # Remove the last two numbers before "all"
    new_parts = parts[:-3] + parts[-1:]
    
    # Reassemble the string
    new_s = '_'.join(new_parts)
    
    return new_s

def get_data_config(s):
    # Split the string into parts
    parts = s.split('_')
 
    
    seedmethod = parts[2]+','+parts[1]
    bcv=parts[-3]
    dropout=parts[-2]
    return seedmethod, bcv, dropout

res_list = []
for run in sweep.runs:
    res, plots, cfg = get_results(run)
    res_list.append(
        dict(
            run_id=run.id,
            res=res,
            plots=plots,
            cfg=cfg
        )
    )

metric_res = []
for i in tqdm(range(len(res_list))):
    print(i)
    datatrue = np.load("../synthetic_data/" + rename_string(res_list[i]['res']['data']) + '.npz')
    datatrue_train = datatrue['data'][datatrue['is_train']]
    datatrue_test = datatrue['data'][~datatrue['is_train']]
    phate_train = res_list[i]['plots']['phate_true_train']
    phate_test = res_list[i]['plots']['phate_true']
    our_train = res_list[i]['plots']['phate_pred_train']
    our_test = res_list[i]['plots']['phate_pred']
    demap_phate_train = demap.DEMaP(datatrue_train, phate_train)
    demap_our_train = demap.DEMaP(datatrue_train, our_train)
    demap_phate_test = demap.DEMaP(datatrue_test, phate_test)
    demap_our_test = demap.DEMaP(datatrue_test, our_test)
    acc_our_train = 1 - res_list[i]['res']['dist_mape_train_train']
    acc_our_test = 1 - res_list[i]['res']['dist_mape_test_test']
    name = res_list[i]['res']['data']
    recon_weights = res_list[i]['res']['dist_recon_weight']
    seedmethod, bcv, dropout = get_data_config(res_list[i]['res']['data'])
    metric_res.append(dict(
        dataset=seedmethod,
        bcv=bcv,
        dropout=dropout,
        recon_weights=recon_weights,
        acc_our_train=acc_our_train,
        acc_our_test=acc_our_test,
        demap_phate_train=demap_phate_train,
        demap_our_train=demap_our_train,
        demap_our_test=demap_our_test,
    ))
    del datatrue, datatrue_train, datatrue_test, phate_train, phate_test, our_train, our_test, demap_phate_train, demap_our_train, demap_phate_test, demap_our_test, acc_our_train, acc_our_test, name, seedmethod, bcv, dropout

res_df = pd.DataFrame(metric_res)

res_df.to_csv("synth_results_many.csv", index=False)

res_df = res_df.sort_values(['dataset', 'bcv', 'dropout'])
# Round all numeric columns to 3 decimals, excluding strings
rounded_res_df = res_df.select_dtypes(include=['float64']).round(3)
# Re-attach the non-numeric columns to the rounded DataFrame
for col in res_df.select_dtypes(exclude=['float64']).columns:
    rounded_res_df[col] = res_df[col]

# Reorder columns to match original DataFrame
rounded_res_df = rounded_res_df[res_df.columns]
rounded_res_df.to_csv("synth_results_many_rounded.csv", index=False)