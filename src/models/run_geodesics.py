'''
    Fetch Autoencoder, WDiscriminator from the pre-trained models,
    Then run geodesics.

'''
import argparse
import wandb
import sys
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from omegaconf import OmegaConf
import shutil

import numpy as np
import os
import glob
import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from plotly3d.plot import scatter, trajectories

sys.path.append('../../src/')
from model2 import Autoencoder, Preprocessor, WDiscriminator
from off_manifolder import offmanifolder_maker
from geodesic import jacobian, velocity, CondCurve, GeodesicBridgeOverfit, GeodesicBridge


def fetch_model(sweep_id, data_name, model_type='autoencoder', weights_cycle=1, dimensions_latent=3, clamp=0.1):
    wandb.login()
    api = wandb.Api()

    entity = "xingzhis"
    project = "dmae"
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    # Initialize an empty list to store run data
    runs_data = []

    # Iterate through each run in the sweep
    for run in sweep.runs:
        # Extract metrics and configs
        metrics = run.summary._json_dict
        configs = run.config
        
        # Combine metrics and configs, and add run ID
        combined_data = {**metrics, **configs, "run_id": run.id}
        
        # Append the combined data to the list
        runs_data.append(combined_data)

    # Create a DataFrame from the runs data
    df = pd.DataFrame(runs_data)
    print('Fetched Run ID: ', df['run_id'])
    if model_type == 'autoencoder':
        run_ids = df[(df['data.name'] == data_name) & \
                    (df['loss.weights.cycle'] == weights_cycle) & (df['dimensions.latent'] == dimensions_latent)]['run_id']
    elif model_type == 'wdiscriminator':
        print(df.head())
        run_ids = df[(df['data.name'] == data_name) & (df['cfg/training/clamp'] == clamp)]['run_id']

    assert len(run_ids) == 1
    run_id = run_ids.iloc[0]
    print('Located RunID: ', run_id)
    run = api.run(f"{entity}/{project}/{run_id}")

    # Load the model.
    folder_path = '/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/src/wandb/'
    cfg = OmegaConf.create(run.config)
    folder_list = glob.glob(f"{folder_path}*{run.id}*")
    ckpt_files = glob.glob(f"{folder_list[0]}/files/*.ckpt")
    ckpt_path = ckpt_files[0]

    if model_type == 'autoencoder':
        model = Autoencoder.load_from_checkpoint(ckpt_path)
    elif model_type == 'wdiscriminator':
        model = WDiscriminator.load_from_checkpoint(ckpt_path)
    else:
        raise ValueError(f"model_type {model_type} not recognized.")

    return model, cfg


def load_data(cfg):
    path_root = "/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/src"
    data = np.load(f"{path_root}/{cfg.data.root}/{cfg.data.name}{cfg.data.filetype}", allow_pickle=True)

    return data

def fit_geodesics(model, wd, x, probab, starts, ends, ts):
    '''
        Fit a geodesic between start and end in ambient space.
    Args:
        model: Autoencoder model.
        wd: WDiscriminator model.
        x: np.dnarray of shape [N, d] ambient space points.
        probab: torch.tensor of shape [N, 1] probability of being on the manifold.

        starts, ends: np.dnarray of shape [N, d]
        ts: torch.tensor of shape [N, 1] time points where to evaluate the geodesic.
    Returns:
        geodesic: torch.tensor of shape [N, len(ts), d] geodesic points.
    '''

    print('Fitting geodesics...')

    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=probab.dtype, device=probab.device)

    xbatch = torch.tensor(starts, dtype=x.dtype, device=x.device)
    xendbatch = torch.tensor(ends, dtype=x.dtype, device=x.device)

    ids = torch.zeros((xbatch.size(0),1))

    dataset = TensorDataset(xbatch, xendbatch, ids)
    dataloader = DataLoader(dataset, batch_size=len(x), shuffle=True)

    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in wd.parameters():
        param.requires_grad = False
    enc_func = lambda x: model.encoder(x)
    disc_func = lambda x: (wd(x).flatten()-probab.min())/(probab.max()-probab.min()) # Min-max normalization to [0,1].

    ofm = offmanifolder_maker(enc_func, disc_func, disc_factor=0.5, max_prob=probab.max())
    gbmodel = GeodesicBridgeOverfit(
        func=ofm,
        input_dim=x.size(1), 
        hidden_dim=32,
        scale_factor=1, 
        symmetric=True,
        num_layers=2, 
        n_tsteps=len(ts), 
        lr=1e-4, 
        weight_decay=1e-3,
        discriminator_weight=0.,
        discriminator_func_for_grad_weight=0.,
        id_dim=1,
        id_emb_dim=1,
        density_weight=0.,
        length_weight=1.,
    )
    
    gbmodel.lr=1e-3
    trainer = pl.Trainer(
        max_epochs=400,
        log_every_n_steps=20,
        accelerator='auto',
    )
    trainer.fit(gbmodel, dataloader)

    print('Done fitting geodesics.')

    return gbmodel


if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser(description='Run geodesics.')
    parser.add_argument('--ae_sweep_id', type=str, default='59muwbsf', help='Sweep ID of the Autoencoder model.')
    parser.add_argument('--disc_sweep_id', type=str, default='g0066w8g', help='Sweep ID of the WDiscriminator model.')
    parser.add_argument('--data_name', type=str, default='hemisphere_none_0', help='Name of the data.')
    parser.add_argument('--weights_cycle', type=float, default=1., help='Cycle of the weights.')
    parser.add_argument('--dimensions_latent', type=int, default=3, help='Number of latent dimensions.')
    parser.add_argument('--clamp', type=float, default=0.1, help='Clamp value for the WDiscriminator.')
    args = parser.parse_args()

    ae_sweep_id = args.ae_sweep_id
    disc_sweep_id = args.disc_sweep_id
    data_name = args.data_name
    weights_cycle = args.weights_cycle
    dimensions_latent = args.dimensions_latent
    clamp = args.clamp
    
    # Load Autoencoder model.
    print('Loading Autoencoder model...')
    ae_model, ae_cfg = fetch_model(sweep_id=ae_sweep_id, data_name=data_name, model_type='autoencoder', weights_cycle=1, dimensions_latent=3)

    # Load data.
    data = load_data(ae_cfg)

    # Load WDiscriminator model.
    print('Loading WDiscriminator model...')
    wd_model, wd_cfg = fetch_model(sweep_id=disc_sweep_id, data_name=data_name, model_type='wdiscriminator', clamp=clamp)

    # Run geodesics.
    with torch.no_grad():
        wd_model.eval()
        x_and_neg = torch.tensor(data['data'], dtype=torch.float32, device=wd_model.device)
        probab = wd_model(x_and_neg).flatten()
        z = ae_model.encoder(x_and_neg)

    t_steps = 100
    ts = torch.linspace(0, 1, t_steps)
    geo_model = fit_geodesics(ae_model, wd_model, 
                              data['data'], probab, 
                              data['start_points'], data['end_points'], ts)
    
    # Save the geodesic model.
    save_folder = './geomodels_pretrained'
    if os.path.exists(save_folder):
        # remove the folder and its contents
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    torch.save(geo_model.state_dict(), f"{save_folder}/geo_model.ckpt")

    print(f'Geodesic model saved to {save_folder} .')

    # Plot the geodesics.
    starts = data['start_points']
    ends = data['end_points']
    xbatch = torch.tensor(starts, dtype=z.dtype, device=z.device)
    xendbatch = torch.tensor(ends, dtype=z.dtype, device=z.device)
    ids = torch.zeros((xbatch.size(0),1), device=z.device)

    pred_geodesics = None
    with torch.no_grad():
        geo_model.eval()
        pred_geodesics = geo_model(xbatch, xendbatch, ts, ids)

    # Plot X, neg_X, and probability.
    X = data['data']
    #neg_X = data['data'][~data['mask_x']]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2], mode='markers', 
                               marker=dict(size=3, color=probab, colorscale='Viridis')))

    # Plot starts, ends, geodesics.
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2], mode='markers', marker=dict(size=3, color='gray')))
    fig.add_trace(go.Scatter3d(x=starts[:,0], y=starts[:,1], z=starts[:,2], mode='markers', marker=dict(size=3, color='red')))
    fig.add_trace(go.Scatter3d(x=ends[:,0], y=ends[:,1], z=ends[:,2], mode='markers', marker=dict(size=3, color='blue')))
    for i in range(starts.shape[0]):
        fig.add_trace(go.Scatter3d(x=pred_geodesics[i,:,0], y=pred_geodesics[i,:,1], z=pred_geodesics[i,:,2], mode='lines', line=dict(width=2)))
    fig.write_html(f'{save_folder}/geodesic.html')


