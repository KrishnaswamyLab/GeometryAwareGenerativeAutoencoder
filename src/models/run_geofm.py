'''
    Fetch Autoencoder, WDiscriminator from the pre-trained models,
    Then run Geodesic Flow Matching.

    Run geodesic flow matching
    1. Define x0, x1 start/end points on ambient data space.
    2. Create off_manifolder by passing encoder to encode points both on/off manifold, thus learn a meaningful metric
    3. Pass 1)off_manifolder and 2)encoder to GeodesicFM model. Freeze off_manifolder's and encoder's parameters. Encoder is needed for OPT.
    4. Train GeodesicFM and infer.
'''
from typing import Tuple
import argparse
import os
import sys
import numpy as np
import torch
import shutil
import pytorch_lightning as pl
import wandb
import plotly.graph_objects as go
from omegaconf import OmegaConf
from glob import glob
from torch.utils.data import Dataset, DataLoader

sys.path.append('../../src/')
from model2 import Autoencoder, Discriminator
from off_manifolder import offmanifolder_maker_new
from geodesic import GeodesicFM

import torchdiffeq
from torch import nn

adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def plot_trajectories(Z, z0, z1, trajs, title='Geodesic Flow Paths', save_path=None):
    '''
        Plot the trajectories in latent space.
    Args:
        Z: np.ndarray of shape [N, D] latent space points.
        z0: np.ndarray of shape [B, D] latent embeds of x0.
        z1: np.ndarray of shape [B, D] latent embeds of x1.
        trajs: np.ndarray of shape [T, B, D] latent space trajectory.
        title: str, title of the plot.
    '''
    assert Z.shape[1] == z0.shape[1] == z1.shape[1] == trajs.shape[2]

    print('Plotting Trajectories...')
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=Z[:,0], y=Z[:,1], z=Z[:,2], 
                           mode='markers', marker=dict(size=2, color='gray', colorscale='Viridis', opacity=0.8)))
    fig.add_trace(go.Scatter3d(x=z0[:,0], y=z0[:,1], z=z0[:,2],
                                mode='markers', marker=dict(size=5, color='blue', opacity=0.8)))
    fig.add_trace(go.Scatter3d(x=z1[:,0], y=z1[:,1], z=z1[:,2],
                                mode='markers', marker=dict(size=5, color='green', opacity=0.8)))

    for i in range(trajs.shape[1]):
        fig.add_trace(go.Scatter3d(x=trajs[:,i,0], y=trajs[:,i,1], z=trajs[:,i,2],
                                mode='lines', line=dict(width=2, color='blue')))
    
    fig.update_layout(title=title)
    
    if save_path is not None:
        fig.write_html(save_path)


def fetch_model(model_type='autoencoder', sweep_id='59muwbsf', data_name='eb', 
                weights_cycle=1, dimensions_latent=3, clamp=0.1,
                folder_path='/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/src/wandb/',
                run_id=None):
    if run_id is None:
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
    else:
        print('Using RunID: ', run_id)

    # Load the model.
    #folder_path = '/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/src/wandb/'
    
    folder_list = glob(f"{folder_path}*{run_id}*")
    ckpt_files = glob(f"{folder_list[0]}/files/*.ckpt")
    ckpt_path = ckpt_files[0]

    if model_type == 'autoencoder':
        model = Autoencoder.load_from_checkpoint(ckpt_path)
    elif model_type == 'wdiscriminator':
        model = Discriminator.load_from_checkpoint(ckpt_path)
    else:
        raise ValueError(f"model_type {model_type} not recognized.")
    
    cfg = model.hparams.cfg

    return model, cfg

def fetch_local_model(model_type='autoencoder', ckpt_path=None):
    if model_type == 'autoencoder':
        model = Autoencoder.load_from_checkpoint(ckpt_path)
    elif model_type == 'wdiscriminator':
        model = Discriminator.load_from_checkpoint(ckpt_path)
    else:
        raise ValueError(f"model_type {model_type} not recognized.")
    
    cfg = model.hparams.cfg

    return model, cfg


def load_data(cfg, path_root="/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/src"):
    data = np.load(f"{path_root}/{cfg.data.root}/{cfg.data.name}{cfg.data.filetype}", allow_pickle=True)

    return data

class CustomDataset(Dataset):
    def __init__(self, x0, x1):
        self.x0 = x0
        self.x1 = x1

    def __len__(self):
        return max(len(self.x0), len(self.x1))

    def __getitem__(self, idx):
        return self.x0[idx % len(self.x0)], self.x1[idx % len(self.x1)]

def custom_collate_fn(batch):
    x0_batch = torch.stack([item[0] for item in batch])
    x1_batch = torch.stack([item[1] for item in batch])
    
    # Randomly permute the elements in the batch
    perm_x0 = torch.randperm(len(x0_batch))
    perm_x1 = torch.randperm(len(x1_batch))

    x0_batch = x0_batch[perm_x0]
    x1_batch = x1_batch[perm_x1]
    
    return x0_batch, x1_batch

def sample_indices_within_range(points, selected_idx=None, range_size=0.1, num_samples=20, seed=23):
    '''
        Randomly sample points within a range of two selected points.
    Args:
        points: np.ndarray of shape [N, d] ambient space points.
        selected_idx: tuple of two indices of points.
        range_size: float, range size.
        num_samples: int, number of samples.
        seed: int, random seed.
    Returns:
        point1_idx: int, index of the first point.
        sampled_indices_point1: np.ndarray of shape [num_samples,] indices of sampled points around p1.
        point2_idx: int, index of the second point.
        sampled_indices_point2: np.ndarray of shape [num_samples,] indices of sampled points around p2.
    '''
    np.random.seed(seed)
    # Randomly select two points from the array
    if selected_idx is None:
        selected_indices = np.random.choice(points.shape[0], 2, replace=False)
        point1_idx, point2_idx = selected_indices[0], selected_indices[1]
        point1, point2 = points[point1_idx], points[point2_idx]
    else:
        point1_idx, point2_idx = selected_idx
        point1, point2 = points[point1_idx], points[point2_idx]    
    # Function to find indices of points within the range of a given point
    def _find_indices_within_range(point):
        distances = np.linalg.norm(points - point, axis=1)
        within_range_indices = np.where(distances <= range_size)[0]
        return within_range_indices
    
    # Find indices within range of point1 and point2
    indices_within_range1 = _find_indices_within_range(point1)
    indices_within_range2 = _find_indices_within_range(point2)
    
    # Randomly sample indices within the range
    if len(indices_within_range1) >= num_samples:
        sampled_indices_point1 = np.random.choice(indices_within_range1, num_samples, replace=False)
    else:
        sampled_indices_point1 = indices_within_range1
    
    if len(indices_within_range2) >= num_samples:
        sampled_indices_point2 = np.random.choice(indices_within_range2, num_samples, replace=False)
    else:
        sampled_indices_point2 = indices_within_range2
    
    return point1_idx, sampled_indices_point1, point2_idx, sampled_indices_point2


def get_geobridge_path(gbmodel, x0, x1, ae_model, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
        Directly get the geobridge path from the trained flow matching model.
    Args:
        gbmodel: GeodesicFM model.
        x0: np.ndarray of shape [N, d] ambient space points.
        x1: np.ndarray of shape [N, d] ambient space points.
        ae_model: Autoencoder model.
        device: str, device type.
    Returns:
        z_traj: np.ndarray of shape [T, B, D] latent space trajectory.
        z0: np.ndarray of shape [B, D] latent embeds of x0.
        z1: np.ndarray of shape [B, D] latent embeds of x1.
    '''
    if isinstance(x0, np.ndarray):
        x0 = torch.tensor(x0, dtype=torch.float32, device=device)
        x1 = torch.tensor(x1, dtype=torch.float32, device=device)

    ids = torch.zeros((x0.size(0),1)) # Conditional id for each pair of x0 and x1, here is a dummy.
    ids = ids.to(device)

    gbmodel = gbmodel.to(device)
    gbmodel.eval()
    with torch.no_grad():
        x_traj = gbmodel(x0, x1, gbmodel.ts.to(device), ids) # [T, B, D]
    print('Predicted trajectory shape: ', x_traj.shape)

    ae_model.eval()
    with torch.no_grad():
        z_traj = ae_model.encoder(x_traj.flatten(0,1)) # [T*B, D]

        z0 = ae_model.encoder(x0)
        z1 = ae_model.encoder(x1)

    z_traj = z_traj.cpu().detach().numpy()
    z_traj = z_traj.reshape(x_traj.size(0), x_traj.size(1), -1)
    print('z_traj.shape: ', z_traj.shape)

    z0 = z0.cpu().detach().numpy()
    z1 = z1.cpu().detach().numpy()

    return z_traj, z0, z1


class ODEFuncWrapper(nn.Module):
    def __init__(self, flowmodel):
        super().__init__()
        self.flowmodel = flowmodel
    def forward(self, t, y):
        # Expand t to match the batch size and feature dimension
        t_expanded = t.view(1, 1).expand(y.size(0), 1)
        # Concatenate y and t along the feature dimension
        y_with_t = torch.cat((y, t_expanded), dim=-1)
        return self.flowmodel(y_with_t)
    
def infer_flow_paths(gbmodel, x0, sampled_indices_point1, ae_model, device) -> Tuple[np.ndarray, np.ndarray]:
    '''
        Infer Trajectories using ODE on vector fields predicted by FM.
    Args:
        gbmodel: GeodesicFM model.
        x0: np.ndarray of shape [N, d] ambient space points.
        sampled_indices_point1: np.ndarray of shape [n_samples,] indices of sampled points.
        ae_model: Autoencoder model.
        device: str, device type.
    Returns:
        z_traj: np.ndarray of shape [T, B, D] latent space trajectory.
        sampled_starts: np.ndarray of shape [n_samples, d] sampled start points.
    '''

    n_samples = x0.shape[0]
    sampled_starts = torch.tensor(x[sampled_indices_point1[:n_samples]], dtype=torch.float32).to('cpu')

    print(f'Run ODE on {n_samples} samples: {sampled_starts.shape}')
    gbmodel.eval()
    flowfunc = ODEFuncWrapper(gbmodel.flow_model.to('cpu'))
    with torch.no_grad():
        traj = odeint(flowfunc, sampled_starts.to('cpu'), gbmodel.ts.to('cpu'))

    print('Flow Matching ODE Trajectory shape: ', traj.shape)

    ae_model.eval()
    with torch.no_grad():
        traj = traj.to(device)
        z_traj = ae_model.encoder(traj.flatten(0,1)) # [T*B, D]

    z_traj = z_traj.cpu().detach().numpy()
    z_traj = z_traj.reshape(traj.size(0), traj.size(1), -1)
    print('z_traj.shape: ', z_traj.shape)

    return z_traj, sampled_starts


def fit_flow_matching(model, wd, x, starts, ends, n_tsteps=100,
                      max_epoch=400, lr=1e-3, weight_decay=1e-3):
    '''
        Fit a geodesic between start and end in ambient space.
    Args:
        model: Autoencoder model.
        wd: WDiscriminator model.
        x: np.dnarray of shape [N, d] ambient space points.

        starts, ends: np.dnarray of shape [N, d]
        n_tsteps: int, number of time steps.
    Returns:
        geodesic: torch.tensor of shape [len(ts), N, d] geodesic points.
    '''

    print('Fitting Geodesics Flow Matching ...')

    # Create a dataloader
    dataset = CustomDataset(x0=torch.tensor(starts, dtype=torch.float32, device=model.device),
                            x1=torch.tensor(ends, dtype=torch.float32, device=model.device))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32, device=model.device)

    # Freeze the encoder and WDiscriminator parameters; 
    # Make OffManioflder that penalizes points off the manifold.
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in wd.parameters():
        param.requires_grad = False
    model.eval()
    wd.eval()
    wd.to(model.device)
    enc_func = lambda x: model.encoder(x)
    disc_func = lambda x: 1 - wd.positive_proba(x)
    ofm, _ = offmanifolder_maker_new(enc_func, disc_func, disc_factor=8) # ofm encodes both on/off manifold points
    
    # Fit Geodesic Flow Matching.
    gbmodel = GeodesicFM(
        func=ofm,
        encoder=enc_func,
        input_dim=x.shape[1],
        hidden_dim=64,
        scale_factor=1, 
        symmetric=True, 
        num_layers=3, 
        n_tsteps=n_tsteps, 
        lr=lr,
        weight_decay=weight_decay,
    )
    gbmodel.to(model.device)
    
    gbmodel.lr=1e-3
    trainer = pl.Trainer(
        max_epochs=max_epoch,
        log_every_n_steps=20,
        accelerator='auto',
    )

    trainer.fit(gbmodel, dataloader)

    print('Done fitting Geodeisc Flow Matching.')

    return gbmodel


if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser(description='Run Geodesics Flow Matching.')
    parser.add_argument('--max_epoch', type=int, default=400, help='Number of epochs to train the model.')
    parser.add_argument('--ae_sweep_id', type=str, default='59muwbsf', help='Sweep ID of the Autoencoder model.')
    parser.add_argument('--disc_sweep_id', type=str, default='g0066w8g', help='Sweep ID of the WDiscriminator model.')
    parser.add_argument('--data_name', type=str, default='eb', help='Name of the data.')
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
    ae_model, ae_cfg = fetch_model(model_type='autoencoder', run_id='pzlwi6t6', folder_path='../wandb/')

    # Load data.
    data = load_data(ae_cfg, path_root='../')

    # Load Discriminator model.
    print('Loading Discriminator model...')
    wd_model, wd_cfg = fetch_model(model_type='wdiscriminator', run_id='kafcutw4', folder_path='../wandb/')
   
    # Run encoder to get encodings.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'

    x = data['data']
    batch_size = 32

    encodings = []
    ae_model.to(device)
    ae_model.eval()
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            x_batch = torch.tensor(x[i:i+batch_size], dtype=torch.float32, device=device)
            enc_batch = ae_model.encoder(x_batch)
            encodings.append(enc_batch.cpu().numpy())
    encodings = np.concatenate(encodings, axis=0)
    print('Encodings shape: ', encodings.shape)

    # Sample starts/ends
    start_group = 0
    end_group = 1

    start_idices = np.where(data['colors'] == start_group)[0]
    start_idx = np.random.choice(start_idices, 1)[0]
    end_idices = np.where(data['colors'] == end_group)[0]
    end_idx = np.random.choice(end_idices, 1)[0]
    print('start_idx, end_idx: ', start_idx, end_idx)

    # start_idx = 736 
    # end_idx = 2543

    point1_idx, sampled_indices_point1, point2_idx, sampled_indices_point2 = sample_indices_within_range(encodings, 
                                                                                                        selected_idx=(start_idx, end_idx),
                                                                                                        range_size=0.5, 
                                                                                                        seed=2024, num_samples=50)
    point1 = x[point1_idx]
    point2 = x[point2_idx]
    samples_point1 = x[sampled_indices_point1]
    samples_point2 = x[sampled_indices_point2]

    print('point1, point2: ', point1.shape, point2.shape)
    print('samples_point1, samples_point2: ', samples_point1.shape, samples_point2.shape)

    # Fit Geodesic Flow Matching.
    gbmodel = fit_flow_matching(ae_model, wd_model, x, samples_point1, samples_point2, 
                                n_tsteps=100, max_epoch=args.max_epoch, lr=1e-3, weight_decay=1e-3)
    
    # Save the model.
    path_root = "./fm/"
    os.makedirs(path_root, exist_ok=True)
    os.makedirs(f"{path_root}/ckpt", exist_ok=True)
    torch.save(gbmodel.state_dict(), f"{path_root}/ckpt/geodesic_flow_matching.ckpt")

    # Infer geodesic bridge path.
    print('Getting geodesic bridge path...')
    z_traj, z0, z1 = get_geobridge_path(gbmodel, samples_point1, samples_point2, ae_model, device)
    plot_trajectories(encodings, z0, z1, z_traj, title='Geodesic Bridge Paths', save_path=f"{path_root}/geodesic_bridge_paths.html")


    # Infer the flow paths.    
    print('Getting ODE flow paths...')
    z_traj, sampled_starts = infer_flow_paths(gbmodel, x, sampled_indices_point1, ae_model, device)
    plot_trajectories(encodings, z0, z1, z_traj, title='Flow Paths', save_path=f"{path_root}/flow_paths.html")

