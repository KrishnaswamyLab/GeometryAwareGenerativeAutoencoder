import wandb
import sys
import matplotlib.pyplot as plt
import scprep
import pandas as pd
import numpy as np
import os
import glob
import torch
from model2 import Autoencoder
from omegaconf import OmegaConf
from geodesic import jacobian
from procrustes import Procrustes
from scipy.stats import gaussian_kde
import math

sys.path.append('../../dmae/src/')

def load_sweep_data(entity, project, sweep_id, folder_path):
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs_data = []

    for run in sweep.runs:
        metrics = run.summary._json_dict
        configs = run.config
        combined_data = {**metrics, **configs, "run_id": run.id}
        
        cfg = OmegaConf.create(run.config)
        folder_list = glob.glob(f"{folder_path}*{run.id}*")
        ckpt_files = glob.glob(f"{folder_list[0]}/files/*.ckpt")
        ckpt_path = ckpt_files[0]
        
        model = Autoencoder.load_from_checkpoint(ckpt_path)
        data = np.load(f"data/{cfg.data.name}{cfg.data.filetype}", allow_pickle=True)

        with torch.no_grad():
            model.eval()
            x = torch.tensor(data['data'], dtype=torch.float32, device=model.device)
            z = model.encoder(x).cpu().numpy()

        runs_data.append(combined_data)

    return pd.DataFrame(runs_data)

def get_run_id(df, dim, dataset, data_type, method):
    filtered_df = df[
        (df['cfg/dimensions/latent'] == dim) & 
        (df['cfg/data/name'] == f"{dataset}_{data_type}_{method}")
    ]
    
    if len(filtered_df) == 0:
        raise ValueError("No matching runs found.")
    elif len(filtered_df) == 1:
        return filtered_df.iloc[0]['run_id']
    else:
        return filtered_df.loc[filtered_df['validation/dist_loss'].idxmin()]['run_id']

def area_element(u, v, dataset):
    if dataset == 'saddle':
        E = 1 + (2 * u)**2
        F = (2 * u) * (-2 * v)
        G = 1 + (2 * v)**2
        dA = np.sqrt(E * G - F**2)
    elif dataset == 'hemishpere':
        dA = 1 / (1 - u**2 - v**2)
    elif dataset == 'paraboloid':
        E = 1 + (2 * u)**2  # E = 1 + 4u^2
        F = (2 * u) * (2 * v)  # F = 4uv
        G = 1 + (2 * v)**2  # G = 1 + 4v^2
        dA = np.sqrt(E * G - F**2)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return dA

def area_element_torch(u, v, dataset):
    if dataset == 'saddle':
        E = 1 + (2 * u)**2
        F = (2 * u) * (-2 * v)
        G = 1 + (2 * v)**2
        dA = torch.sqrt(E * G - F**2)
    elif dataset == 'hemishpere':
        dA = 1 / (1 - u**2 - v**2)
    elif dataset == 'paraboloid':
        E = 1 + (2 * u)**2  # E = 1 + 4u^2
        F = (2 * u) * (2 * v)  # F = 4uv
        G = 1 + (2 * v)**2  # G = 1 + 4v^2
        dA = torch.sqrt(E * G - F**2)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return dA

def kde(data, points_to_evaluate=None, bandwidth=None):
    is_torch = isinstance(data, torch.Tensor)
    
    if is_torch:
        original_dtype = data.dtype
        original_device = data.device
        data_np = data.cpu().numpy()
    else:
        data_np = data
    
    kde = gaussian_kde(data_np.T, bw_method=bandwidth)
    
    if points_to_evaluate is None:
        points_to_evaluate = data_np
    elif is_torch and isinstance(points_to_evaluate, torch.Tensor):
        points_to_evaluate = points_to_evaluate.cpu().numpy()
    
    result = kde(points_to_evaluate.T)
    
    if is_torch:
        result = torch.tensor(result, dtype=original_dtype, device=original_device)
    
    return result

def main(dim, dataset, data_type, method, save_path):
    entity = "xingzhis"
    project = "dmae"
    sweep_id = 'b8g2xpjd'
    folder_path = '../../src/wandb/'

    # Create save_path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    df = load_sweep_data(entity, project, sweep_id, folder_path)
    run_id = get_run_id(df, dim, dataset, data_type, method)

    print(f"Selected run_id: {run_id}")

    run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    cfg = OmegaConf.create(run.config)
    folder_list = glob.glob(f"{folder_path}*{run.id}*")
    ckpt_files = glob.glob(f"{folder_list[0]}/files/*.ckpt")
    ckpt_path = ckpt_files[0]

    model = Autoencoder.load_from_checkpoint(ckpt_path)
    data = np.load(f"data/{cfg.data.name}{cfg.data.filetype}", allow_pickle=True)

    with torch.no_grad():
        model.eval()
        x = torch.tensor(data['data'], dtype=torch.float32, device=model.device)
        z = model.encoder(x)
        xh = model.decoder(z)
        z = z.cpu().numpy()
        xh = xh.cpu().numpy()

    # proc = Procrustes()
    # p1, z1, di = proc.fit_transform(data['phate'], z)
    z1 = z

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    scprep.plot.scatter2d(data['phate'], ax=axs[0, 0], title='Original PHATE')
    scprep.plot.scatter2d(z1, ax=axs[0, 1], title='Encoded Space')

    ax1 = fig.add_subplot(223, projection='3d')
    scprep.plot.scatter3d(data['data'], ax=ax1, title='Original Data')

    ax2 = fig.add_subplot(224, projection='3d')
    scprep.plot.scatter3d(xh, ax=ax2, title='Reconstructed Data')

    plt.suptitle(f'{dataset.capitalize()}-{data_type.capitalize()}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{dataset}_{data_type}_{method}_dim{dim}_comparison.png'))
    plt.close()

    # Jacobian and volume element calculations
    jac = jacobian(model.encoder, x)
    U, S, Vh = torch.linalg.svd(jac, full_matrices=False)
    vol_elem = S.prod(axis=1)

    vol_elem_analycial = area_element_torch(x[:,0], x[:,1], dataset)

    kde_x = kde(x)
    kde_uv = kde(x[:, :2])

    # Plotting volume element comparisons
    no_margin_mask = (x[:,0]>-1.5) & (x[:,0]<1.5) & (x[:,1]>-1.5) & (x[:,1]<1.5)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(221)
    plt.scatter((vol_elem).detach().numpy()[no_margin_mask], (vol_elem_analycial).detach().numpy()[no_margin_mask])
    plt.title("Volume Element Comparison")
    
    plt.subplot(222)
    plt.scatter(torch.log(vol_elem).detach().numpy()[no_margin_mask], torch.log(vol_elem_analycial).detach().numpy()[no_margin_mask])
    plt.title("Log Volume Element Comparison")
    
    plt.subplot(223)
    plt.scatter(torch.sqrt(vol_elem).detach().numpy()[no_margin_mask], 1/(vol_elem_analycial).detach().numpy()[no_margin_mask])
    plt.title("Sqrt Vol Elem vs 1/Analytical Vol Elem")
    
    plt.subplot(224)
    plt.scatter((vol_elem).detach().numpy()[no_margin_mask], 1/(vol_elem_analycial).detach().numpy()[no_margin_mask])
    plt.title("Vol Elem vs 1/Analytical Vol Elem")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{dataset}_{data_type}_{method}_dim{dim}_volume_element_comparison.png'))
    plt.close()

    # Additional plots
    threshold = 1e-2
    mask = (kde_x>threshold) & (kde_uv>threshold)
    plt.scatter((vol_elem/kde_x).detach().numpy()[mask], (vol_elem_analycial/kde_uv).detach().numpy()[mask])
    plt.savefig(os.path.join(save_path, f'{dataset}_{data_type}_{method}_dim{dim}_kde_comparison.png'))
    plt.close()

    mask = (kde_x>threshold) & (kde_uv>threshold) & (vol_elem>threshold) & (vol_elem_analycial>threshold)
    plt.scatter(1/(vol_elem/kde_x).detach().numpy()[mask], (vol_elem_analycial/kde_uv).detach().numpy()[mask])
    plt.savefig(os.path.join(save_path, f'{dataset}_{data_type}_{method}_dim{dim}_inverse_kde_comparison.png'))
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process data and visualize results.")
    parser.add_argument("--dim", type=int, default=2, help="The latent dimension.")
    parser.add_argument("--dataset", default="saddle", help="The dataset name (e.g., 'saddle', 'hemishpere', 'paraboloid').")
    parser.add_argument("--data_type", default="uniform", choices=["uniform", "gaussian"], help="The type of data distribution.")
    parser.add_argument("--method", default="heatgeo", help="The method used (e.g., 'heatgeo').")
    parser.add_argument("--save_path", default="./results", help="Path to save the results.")
    args = parser.parse_args()

    main(args.dim, args.dataset, args.data_type, args.method, args.save_path)