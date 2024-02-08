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
    data_path = os.path.join(cfg.data.root, cfg.data.name + "_all" + cfg.data.filetype)
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
    test_all_mask = test_mask[:,None] * np.ones_like(test_mask)
    eps = 1e-10
    dist_mape_test_test = (np.abs(dist_true - dist_pred + eps) / (dist_true + eps) * test_test_mask).sum() / test_test_mask.sum()
    dist_mape_test_train = (np.abs(dist_true - dist_pred + eps) / (dist_true + eps) * test_train_mask).sum() / test_train_mask.sum()
    dist_mape_test_overall = (np.abs(dist_true - dist_pred + eps) / (dist_true + eps) * test_all_mask).sum() / test_all_mask.sum()
    dist_rmse_test_test = np.sqrt(((dist_true - dist_pred)**2 * test_test_mask).sum()/ test_test_mask.sum())
    dist_rmse_test_train = np.sqrt(((dist_true - dist_pred)**2 * test_train_mask).sum() / test_train_mask.sum())
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
        dist_rmse_test_test=dist_rmse_test_test,
        dist_rmse_test_train=dist_rmse_test_train,
        test_rmse=test_rmse,
    )
    plot_data = dict(
        phate_true = data_all['phate'][test_mask],
        phate_pred = zhat_all[test_mask],
        colors = data_all['colors'][test_mask],
        dist_true_test_test = dist_true[test_mask][:,test_mask],
        dist_pred_test_test = dist_pred[test_mask][:,test_mask],
        dist_true_test_train = dist_true[test_mask][:,train_mask],
        dist_pred_test_train = dist_pred[test_mask][:,train_mask],
    )
    return res, plot_data, cfg

