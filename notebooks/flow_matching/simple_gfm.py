import argparse
import os
import sys
from glob import glob

import phate
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import scipy.spatial as spatial

sys.path.append('../../src/')
from diffusionmap import DiffusionMap
from negative_sampling import make_hi_freq_noise
from model2 import Autoencoder as OldAutoencoder
from model2 import Discriminator as OldDiscriminator
from train_autoencoder import Autoencoder as LocalAutoencoder
from off_manifolder import offmanifolder_maker_new
from geodesic import GeodesicFM
from train_autoencoder import PointCloudDataset, make_custom_collate_fn, split_pointcloud, Autoencoder

import ot
adjoint = False
if adjoint:
        from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, layer_widths=[64, 64, 64], activation='relu', 
                 batch_norm=False, dropout=0.0, use_spectral_norm=False):
        super().__init__()

        layers = []
        for i, width in enumerate(layer_widths):
            if i == 0:
                linear_layer = torch.nn.Linear(in_dim, width)
            else:
                linear_layer = torch.nn.Linear(layer_widths[i-1], width)

            # Conditionally apply spectral normalization
            if use_spectral_norm:
                linear_layer = spectral_norm(linear_layer)

            layers.append(linear_layer)

            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(width))
            
            if activation == 'relu':
                layers.append(torch.nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(torch.nn.LeakyReLU())
            elif activation == 'tanh':
                layers.append(torch.nn.Tanh())
            else:
                raise ValueError(f'Invalid activation function: {activation}')

            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
        
        # Adding the final layer
        final_linear_layer = torch.nn.Linear(layer_widths[-1], out_dim)
        if use_spectral_norm:
            final_linear_layer = spectral_norm(final_linear_layer)
        layers.append(final_linear_layer)

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Discriminator(pl.LightningModule):
    def __init__(self, in_dim, layer_widths=[256, 128, 64], activation='relu', loss_type='bce', normalize=False,
                 data_pts=None, k=5,
                 batch_norm=False, dropout=0.0, use_spectral_norm=False, lr=1e-4, weight_decay=1e-4):
        super().__init__()
        self.in_dim = in_dim if data_pts is None else in_dim + k
        self.mlp = MLP(self.in_dim, 2, layer_widths, activation, batch_norm, dropout, use_spectral_norm)
        self.loss_type = loss_type
        self.normalize = normalize
        
        # hyperparameters for augmenting data with density feautres
        self.data_pts = data_pts
        self.k = int(k)

        self.lr = lr
        self.weight_decay = weight_decay
        
        self.train_step_outs = []
        self.val_step_outs = []
        self.test_step_outs = []
        self.train_ys = []
        self.val_ys = []
        self.test_ys = []
        
        if self.data_pts is not None:
            assert self.data_pts.shape[1] == self.in_dim

        # Save hyperparameters
        self.save_hyperparameters()
    
    def augment_data(self, x):
        dists = torch.cdist(x, self.data_pts)
        topk, _ = torch.topk(dists, k=self.k, dim=1, largest=False, sorted=False)
        return torch.cat([x, topk], dim=1)

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        if self.data_pts is not None:
            x = self.augment_data(x)
        return self.mlp(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.loss_type == 'bce':
            loss = F.cross_entropy(logits, y)
        elif self.loss_type == 'margin':
            score = logits[:, 0]
            loss = -(torch.mean(score[y==1]) - torch.mean(score[y==0]))
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch, batch_idx)
        self.train_step_outs.append(logits)
        self.train_ys.append(y)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch, batch_idx)
        self.val_step_outs.append(logits)
        self.val_ys.append(y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch, batch_idx)
        self.test_step_outs.append(logits)
        self.test_ys.append(y)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self._log_accuracy('train')

    def on_validation_epoch_end(self):
        self._log_accuracy('val')

    def on_test_epoch_end(self):
        self._log_accuracy('test')
    
    def _log_accuracy(self, phase):
        logits = torch.cat(getattr(self, f'{phase}_step_outs'), dim=0)
        true_classes = torch.cat(getattr(self, f'{phase}_ys'), dim=0)
        pred_classes = torch.argmax(logits, dim=1)
        acc = torch.sum(pred_classes == true_classes) / len(true_classes)
        self.log(f'{phase}_acc', acc, on_epoch=True, prog_bar=True)
        getattr(self, f'{phase}_step_outs').clear()
        getattr(self, f'{phase}_ys').clear()
    
    def positive_prob(self, x):
        return F.softmax(self(x), dim=1)[:, 1]
    
    def positive_score(self, x):
        return self(x)[:, 1]
    
    def negative_score(self, x):
        return self(x)[:, 0]

def train_discriminator(x, x_noisy, encoder, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'

    # Dataloader.
    X = torch.cat([x, x_noisy], dim=0)
    Y = torch.cat([torch.ones(x.shape[0]), torch.zeros(x_noisy.shape[0])], dim=0)
    # Convert Y to LongTensor
    Y = Y.long()
    # Split data into train/val/test
    train_idx = int(0.8 * len(X))
    val_idx = int(0.9 * len(X))
    train_data = X[:train_idx], Y[:train_idx]
    val_data = X[train_idx:val_idx], Y[train_idx:val_idx]
    test_data = X[val_idx:], Y[val_idx:]

    train_dataset = torch.utils.data.TensorDataset(*train_data)
    val_dataset = torch.utils.data.TensorDataset(*val_data)
    test_dataset = torch.utils.data.TensorDataset(*test_data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.disc_batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.disc_batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.disc_batch_size, shuffle=False)

    # Encode data for density features.
    data_pts = None
    if args.disc_use_density_features:
        data_pts = encode_data(x, encoder, args.device)

    # Model. 
    # NOTE: Normalize is set to False since we assume latent features are already normalized.
    # TODO: Use density or not.
    model = Discriminator(in_dim=x.shape[1], layer_widths=args.disc_layer_widths, activation='relu', 
                            loss_type='bce', normalize=False,
                            data_pts=data_pts, k=args.disc_density_k,
                            batch_norm=True, dropout=0.5, use_spectral_norm=True, 
                            lr=args.disc_lr, weight_decay=args.disc_weight_decay)
    
    # Trainer.
    early_stop = pl.callbacks.EarlyStopping(monitor='val_loss', patience=args.disc_early_stop_patience, mode='min')
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', 
                                                    dirpath=args.checkpoint_dir, filename='discriminator')
    
    trainer = pl.Trainer(max_epochs=args.disc_max_epochs, log_every_n_steps=args.disc_log_every_n_steps, 
                         callbacks=[early_stop, model_checkpoint],
                         accelerator=device)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Test model.
    trainer.test(ckpt_path="best", dataloaders=test_dataloader)

    # Load best model.
    best_model = Discriminator.load_from_checkpoint(model_checkpoint.best_model_path)
    
    return best_model

def train_autoencoder(pointcloud, distances, labels, args):
    print('[Train Autoencoder] pointcloud: ', pointcloud.shape, 'distances: ', distances.shape, 'labels: ', labels.shape)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    # Split data into train and validation sets
    train_pointcloud, train_distances, train_labels, val_pointcloud, val_distances, val_labels, test_pointcloud, test_distances, test_labels = split_pointcloud(pointcloud, distances, labels)
    train_dataset = PointCloudDataset(train_pointcloud, train_distances)
    val_dataset = PointCloudDataset(val_pointcloud, val_distances)
    test_dataset = PointCloudDataset(test_pointcloud, test_distances)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=make_custom_collate_fn(train_dataset))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=make_custom_collate_fn(val_dataset))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=make_custom_collate_fn(test_dataset))

    # Visualize each split colored by labels, in 3d.
    fig, axs = plt.subplots(3, 2, figsize=(12, 8), subplot_kw={'projection': '3d'})
    for i, (data, cur_labels) in enumerate([(train_pointcloud, train_labels), (val_pointcloud, val_labels), (test_pointcloud, test_labels)]):
        print('data', data.shape, 'cur_labels', cur_labels.shape)
        axs[i, 0].scatter(data[:, 0], data[:, 1], data[:, 2], c=cur_labels, cmap='viridis')
        # axs[i, 1].scatter(phate_coords[:, 0], phate_coords[:, 1], phate_coords[:, 2], c=labels, cmap='viridis')
        axs[i, 0].set_title(f'Pointcloud {i+1} Colored by Labels')
        # axs[i, 1].set_title(f'Phate Coords Colored by Labels')
        # axs[i].axis('off')
    plt.tight_layout()
    plt.savefig(f'{args.plots_save_dir}/AE_pointcloud_splits.png')

    component_wise_normalization = args.ae_component_wise_normalization
    if component_wise_normalization:
        mean = pointcloud.mean(axis=0) # [D]
        std = pointcloud.std(axis=0) # [D]
    else:
        mean = pointcloud.mean() # scalar
        std = pointcloud.std() # scalar
    dist_std = np.std(distances.flatten()) # scalar
    # Init model.
    ae_model = Autoencoder(data_dim=pointcloud.shape[1], latent_dim=args.ae_latent_dim, encoder_layer_width=args.ae_encoder_layer_width, decoder_layer_width=args.ae_decoder_layer_width, activation=args.ae_activation, 
                           batch_norm=args.ae_batch_norm, dropout=args.ae_dropout, use_spectral_norm=args.ae_use_spectral_norm, 
                           mean=mean, std=std, dist_std=dist_std,
                           dist_mse_decay=args.ae_dist_mse_decay, lr=args.ae_lr, weight_decay=args.ae_weight_decay, 
                           weights_dist=args.ae_weights_dist, weights_reconstr=args.ae_weights_reconstr, 
                           weights_cycle=args.ae_weights_cycle, weights_cycle_dist=args.ae_weights_cycle_dist)
    
    # Train.
    early_stop = pl.callbacks.EarlyStopping(monitor='val/loss', patience=args.ae_early_stop_patience, mode='min')
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor='val/loss', save_top_k=1, mode='min', 
                                                    dirpath=args.checkpoint_dir, filename='autoencoder')
    
    trainer = pl.Trainer(max_epochs=args.ae_max_epochs, log_every_n_steps=args.ae_log_every_n_steps, 
                         callbacks=[early_stop, model_checkpoint],
                         accelerator=device)
    trainer.fit(ae_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test model.
    trainer.test(ckpt_path="best", dataloaders=test_loader)

    # Load best model.
    best_ae_model = Autoencoder.load_from_checkpoint(model_checkpoint.best_model_path)

    # Visualize latent space & reconstruction.
    fig, axs = plt.subplots(3, 3, figsize=(12, 8))
    best_ae_model.to(device)
    for i, (data, labels) in enumerate([(train_pointcloud, train_labels), (val_pointcloud, val_labels), (test_pointcloud, test_labels)]):
        z = best_ae_model.encoder(torch.tensor(data, dtype=torch.float32, device=device), normalize=True)
        xhat_unnormalized = best_ae_model.decoder(z, unnormalize=True)
        z = z.detach().cpu().numpy()
        xhat_unnormalized = xhat_unnormalized.detach().cpu().numpy()
        axs[i, 0].scatter(z[:, 0], z[:, 1], z[:, 2], c=labels, cmap='viridis')
        axs[i, 0].set_title(f'Latent Space of Pointcloud {i+1}')
        axs[i, 1].scatter(xhat_unnormalized[:, 0], xhat_unnormalized[:, 1], xhat_unnormalized[:, 2], c=labels, cmap='viridis')
        axs[i, 1].set_title(f'Reconstruction of Pointcloud {i+1}')
        axs[i, 2].scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis')
        axs[i, 2].set_title(f'Ground Truth of Pointcloud {i+1}')
    plt.tight_layout()
    plt.savefig(f'{args.plots_save_dir}/AE_latent_space_reconstruction.png')

    # plotly visualization
    fig = go.Figure()
    for i, (data, labels) in enumerate([(train_pointcloud, train_labels), (val_pointcloud, val_labels), (test_pointcloud, test_labels)]):
        z = best_ae_model.encoder(torch.tensor(data, dtype=torch.float32, device=device), normalize=True)
        xhat_unnormalized = best_ae_model.decoder(z, unnormalize=True)
        z = z.detach().cpu().numpy()
        xhat_unnormalized = xhat_unnormalized.detach().cpu().numpy()
        fig.add_trace(go.Scatter3d(x=z[:, 0], y=z[:, 1], z=z[:, 2], mode='markers', marker=dict(size=2, color=labels, colorscale='viridis'), name=f'Latent Space of {i}'))
    fig.write_html(f'{args.plots_save_dir}/AE_latent_space.html')

    return best_ae_model


def load_autoencoder(run_id, root_dir):
    run_path = os.path.join(root_dir, 'src/wandb/')
    run_path = glob(f"{run_path}/*{run_id}")[0]
    model_path = glob(f"{run_path}/files/*.ckpt")[0]
    return OldAutoencoder.load_from_checkpoint(model_path)

def load_discriminator(run_id, root_dir):
    run_path = os.path.join(root_dir, 'src/wandb/')
    run_path = glob(f"{run_path}/*{run_id}")[0]
    model_path = glob(f"{run_path}/files/*.ckpt")[0]
    return OldDiscriminator.load_from_checkpoint(model_path)

def load_local_autoencoder(checkpoint_path):
    return LocalAutoencoder.load_from_checkpoint(checkpoint_path)

def load_unified_autoencoder(checkpoint_path):
    return Autoencoder.load_from_checkpoint(checkpoint_path)

def load_data(data_path):
    data = np.load(data_path)
    return data['data'], data['dist'], data['colors'], data['phate']

def encode_data(x, encoder, device):
    batch_size = 256
    encodings = []
    encoder.eval()
    encoder.to(device)
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            x_batch = torch.tensor(x[i:i+batch_size], dtype=torch.float32).to(device)
            encodings.append(encoder(x_batch).cpu().detach().numpy())
    return np.concatenate(encodings, axis=0)


def neg_sample_using_additive(x, noise_levels, noise_rate=1, seed=42, noise='gaussian'):
    def _add_noise(data, noise_rate=1, seed=42, noise='gaussian'):
        np.random.shuffle(data)
        np.random.seed(seed)

        noise_rates = np.ones((data.shape[0], 1))
        if noise == 'gaussian':
            noise = np.random.randn(*data.shape)
            data_noisy = data + noise * noise_rate # noise rate is the std of the noise.
        elif noise == 'hi-freq':
            diff_map_op = DiffusionMap(n_components=3, t=3, random_state=seed).fit(data)
            data_noisy = data + make_hi_freq_noise(data, diff_map_op, noise_rate=noise_rates, add_data_mean=False)

        return data_noisy
    
    x_noisy = []
    for (i, noise_level) in enumerate(noise_levels):
        data_noisy = _add_noise(x, noise_rate=noise_level, seed=seed+i, noise=noise)
        x_noisy.extend(data_noisy)
    
    x_noisy = np.array(x_noisy)
    return x_noisy


def neg_sample_using_diffusion(x, ts, num_steps, beta_start, beta_end, seed=42):
    def _forward_diffusion(x0, t, num_steps, beta_start, beta_end, seed=42):
        '''
        Forward diffusion. q(x_t | x_(t-1)) = N(x_t | sqrt(1-beta_t) * x_(t-1), beta_t * I);
        With alpha_bar_t = cumprod(1-beta_t), 
        we have x_t = sqrt(alpha_bar_t) * x_0 + (1-alpha_bar_t) * epsilon_t
        where epsilon_t ~ N(0, 1).
        t has to be an integer, and less than num_steps.
        '''
        np.random.shuffle(x)
        np.random.seed(seed)

        betas = np.linspace(beta_start, beta_end, num_steps) # [beta_0, beta_1, ..., beta_{T-1}]
        alpha_bars = np.cumprod(1-betas)

        x_t = np.sqrt(alpha_bars[t]) * x0 + (1-alpha_bars[t]) * np.random.randn(*x0.shape)
        print('sqrt(alpha_bar_t): ', np.sqrt(alpha_bars[t]), '1-alpha_bar_t: ', 1-alpha_bars[t])

        return x_t

    x_noisy = []
    for (i, t) in enumerate(ts):
        x_t = _forward_diffusion(x, t, num_steps, beta_start, beta_end, seed=seed+i) # (n_samples, n_features)
        x_noisy.extend(x_t)

    x_noisy = np.array(x_noisy)

    return x_noisy

def sample_indices_within_range(x, encoder, device,labels=None, start_group=None, end_group=None, selected_idx=None, 
                                range_size=0.1, num_samples=32, seed=23):
    
    np.random.seed(seed)
    points = encode_data(x, encoder, device)

    if selected_idx[0] is None or selected_idx[1] is None or args.sample_group_points:
        # Randomly select two points from start/end group.
        assert labels is not None and start_group is not None and end_group is not None
        start_idxs = np.where(labels == start_group)[0]
        end_idxs = np.where(labels == end_group)[0]
        point1_idx, point2_idx = np.random.choice(start_idxs, 1)[0], np.random.choice(end_idxs, 1)[0]
        point1, point2 = points[point1_idx], points[point2_idx]
    else:
        # Use selected points.
        assert selected_idx[0] is not None and selected_idx[1] is not None
        point1_idx, point2_idx = selected_idx
        point1, point2 = points[point1_idx], points[point2_idx]
    print('start_idx: ', point1_idx, 'end_idx: ', point2_idx)

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
    
    # Visualize start/end points in latent space.
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], 
                               mode='markers', marker=dict(size=2, color='gray', colorscale='Viridis', opacity=0.8)))
    fig.add_trace(go.Scatter3d(x=points[sampled_indices_point1,0], y=points[sampled_indices_point1,1], z=points[sampled_indices_point1,2], 
                               mode='markers', marker=dict(size=5, color='blue', colorscale='Viridis', opacity=0.8)))
    fig.add_trace(go.Scatter3d(x=points[sampled_indices_point2,0], y=points[sampled_indices_point2,1], z=points[sampled_indices_point2,2], 
                               mode='markers', marker=dict(size=5, color='red', colorscale='Viridis', opacity=0.8)))
    #fig.show()
    return point1_idx, sampled_indices_point1, point2_idx, sampled_indices_point2

def compute_kernel(X: np.array, Y: np.array, sigma: float = 1.0):
    '''
    Compute the Gaussian kernel between two sets of points.
    Adapted from
    https://github.com/professorwug/diffusion_curvature/blob/master/diffusion_curvature/core.py
    Return:
        G: (n_samples_X, n_samples_Y)
    '''
    # Construct the distance matrix.
    D = spatial.distance.cdist(X, Y)
    # Gaussian kernel
    G = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-D**2) / (2 * sigma**2))
    return G

def sampling_rejection(x, x_noisy, method='density', k=20,threshold=0.01):
    '''
    Reject samples that are too close to the original manifold.
    Return:
        a boolean array of shape (n_samples_noisy,), indicating whether each sample is rejected.
    '''
    if method == 'density':
        k = k
        distances = spatial.distance.cdist(x_noisy, x)
        dist_closest = np.partition(distances, k, axis=1)[:, :k]
        dist_mean = np.mean(dist_closest, axis=1)
        return dist_mean <= threshold
    elif method == 'sugar':
        # Use reverse-MAGIC/SUGAR: P_{TxN} is the transition matrix from x_noisy to x
        # 1. P_{TxN}, column normalized, where P_{t,i} is the prob of going from i in x_noisy to j in x.
        # 2. x_noisey_bar = P_{NxT}(Row normalized) * [P_{TxN} * x_noisy]
        # 3. Compare the distance between x_noisy_bar and x_noisy, reject samples that did not change much, since they are on-manifold.
        G_TN = compute_kernel(x, x_noisy)
        P_TN = G_TN / np.sum(G_TN, axis=0, keepdims=True) # column normalized, (n_samples_x, n_samples_noisy)
        G_NT = G_TN.T
        P_NT = G_NT / np.sum(G_NT, axis=1, keepdims=True) # row normalized, (n_samples_noisy, n_samples_x)
        x_noisy_bar = P_NT @ (P_TN @ x_noisy)
        change = np.linalg.norm(x_noisy - x_noisy_bar, axis=1) # (n_samples_noisy,)
        
        return change <= threshold
    else:
        raise ValueError(f"Invalid sampling rejection method: {method}")

class ODEFuncWrapper(nn.Module):
    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model
    def forward(self, t, x):
        '''
        t: (n_samples, 1)
        x: (n_samples, n_features)
        '''
        # Expand t to match the batch size and feature dimension
        t_expanded = t.view(1, 1).expand(x.size(0), 1)  # (n_samples, 1)
        # Concatenate x and t along the feature dimension
        x_with_t = torch.cat((x, t_expanded), dim=-1) # (n_samples, n_features + 1)
        return self.flow_model(x_with_t) # (n_samples, n_features)

def split_train_val_test(x, test_size=0.1, val_size=0.1):
    '''
        Return train, val, test indices, given x.
    '''
    perm = np.random.permutation(len(x))
    test_size = int(len(x) * test_size)
    val_size = int(len(x) * val_size)
    test_idx = perm[:test_size]
    val_idx = perm[test_size:test_size+val_size]
    train_idx = perm[test_size+val_size:]
    return train_idx, val_idx, test_idx

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
    perm_x0 = torch.randperm(len(x0_batch))
    perm_x1 = torch.randperm(len(x1_batch))
    x0_batch = x0_batch[perm_x0]
    x1_batch = x1_batch[perm_x1]
    return x0_batch, x1_batch

def visualize_trajectory(traj, x_encodings, labels, save_path, title, start_pts=None, end_pts=None, plotly=False):
    '''
    Visualize trajectory in latent space.
    Args:
        traj: (n_tsteps, n_samples, n_features)
        x_encodings: (n_samples, n_features)
        labels: (n_samples,)
        save_path: str,
        title: str,
        start_pts: (n_samples, n_features), optional
        end_pts: (n_samples, n_features), optional
    '''
    assert traj.shape[-1] == x_encodings.shape[-1], 'Trajectory and encodings have different feature dimensions.'
    if start_pts is not None:
        assert start_pts.shape[-1] == x_encodings.shape[-1], 'Start points and encodings have different feature dimensions.'
    if end_pts is not None:
        assert end_pts.shape[-1] == x_encodings.shape[-1], 'End points and encodings have different feature dimensions.'
        
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_encodings[:,0], x_encodings[:,1], x_encodings[:,2], c=labels, cmap='viridis', alpha=0.8)

    traj_flat = traj.reshape(-1, traj.shape[-1]) # (n_tsteps * n_samples, n_features)
    ax.scatter(traj_flat[:,0], traj_flat[:,1], traj_flat[:,2], c='blue', alpha=0.8)

    if start_pts is not None:
        ax.scatter(start_pts[:,0], start_pts[:,1], start_pts[:,2], c='green', alpha=0.8)
    if end_pts is not None:
        ax.scatter(end_pts[:,0], end_pts[:,1], end_pts[:,2], c='red', alpha=0.8)
    print('[Visualization] Saving plot to ', save_path)
    plt.savefig(save_path)

    if plotly:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=x_encodings[:,0], y=x_encodings[:,1], z=x_encodings[:,2], 
                                   mode='markers', marker=dict(size=2, color='gray', colorscale='Viridis', opacity=0.8)))
        for i in range(traj.shape[1]):
            fig.add_trace(go.Scatter3d(x=traj[:,i,0], y=traj[:,i,1], z=traj[:,i,2], 
                                       mode='markers', marker=dict(size=5, color='blue', colorscale='Viridis', opacity=0.8)))
        if start_pts is not None:
            fig.add_trace(go.Scatter3d(x=start_pts[:,0], y=start_pts[:,1], z=start_pts[:,2], 
                                       mode='markers', marker=dict(size=5, color='green', colorscale='Viridis', opacity=0.8)))
        if end_pts is not None:
            fig.add_trace(go.Scatter3d(x=end_pts[:,0], y=end_pts[:,1], z=end_pts[:,2], 
                                       mode='markers', marker=dict(size=5, color='red', colorscale='Viridis', opacity=0.8)))
        fig.update_layout(title=title)
        file_basename = os.path.basename(save_path).split('.')[0]
        file_dir = os.path.dirname(save_path)
        print('[Visualization] Saving plotly plot to ', f'{file_dir}/{file_basename}.html')
        fig.write_html(f'{file_dir}/{file_basename}.html')

def visualize_generated(generated_data, real_data, x_encodings, labels, save_path, title, plotly=False):
    '''
    Visualize generated data in latent space.
    Args:
        generated_data: (n_samples, n_features)
        real_data: (n_samples, n_features)
        x_encodings: (n_samples, n_features)
        labels: (n_samples,)
        save_path: str
        title: str
    '''
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_encodings[:,0], x_encodings[:,1], x_encodings[:,2], c='gray', cmap='viridis', alpha=0.8)
    ax.scatter(generated_data[:,0], generated_data[:,1], generated_data[:,2], c='blue', alpha=0.8)
    ax.scatter(real_data[:,0], real_data[:,1], real_data[:,2], c='red', alpha=0.8)
    print('[Visualization] Saving plot to ', save_path)
    plt.savefig(save_path)

    if plotly:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=x_encodings[:,0], y=x_encodings[:,1], z=x_encodings[:,2], 
                                   mode='markers', marker=dict(size=2, color=labels, colorscale='Viridis', opacity=0.8)))
        fig.add_trace(go.Scatter3d(x=generated_data[:,0], y=generated_data[:,1], z=generated_data[:,2], 
                                   mode='markers', marker=dict(size=2, color='blue', colorscale='Viridis', opacity=0.8)))
        fig.add_trace(go.Scatter3d(x=real_data[:,0], y=real_data[:,1], z=real_data[:,2], 
                                   mode='markers', marker=dict(size=2, color='red', colorscale='Viridis', opacity=0.8)))
        fig.update_layout(title=title)
        file_basename = os.path.basename(save_path).split('.')[0]
        file_dir = os.path.dirname(save_path)
        print('[Visualization] Saving plotly plot to ', f'{file_dir}/{file_basename}.html')
        fig.write_html(f'{file_dir}/{file_basename}.html')

def eval_distributions(generated_data, real_data, cost_metric='euclidean'):
    """
        Compute the Wasserstein 1 distancebetween the generated and real data.
        generated_data: [N, latent_dim]
        real_data: [N, latent_dim]
    """
    print('Computing Wasserstein 1 distance... generated: ', generated_data.shape, 'real: ', real_data.shape)

    if not isinstance(generated_data, torch.Tensor):
        generated_data = torch.tensor(generated_data, dtype=torch.float32)
    if not isinstance(real_data, torch.Tensor):
        real_data = torch.tensor(real_data, dtype=torch.float32)

    if cost_metric == 'euclidean':
        # cost_matrix = torch.cdist(generated_data, real_data)
        cost_matrix = ot.dist(generated_data, real_data, metric='euclidean')
    elif cost_metric == 'cosine':
        cost_matrix = 1 - F.cosine_similarity(generated_data, real_data)
    else:
        raise ValueError(f"Unknown cost metric: {cost_metric}")
    
    # Compute the Wasserstein 1 distance
    print('Wasserstein distance Cost Matrix shape: ', cost_matrix.shape)
    # generated_distribution = torch.tensor([1 / generated_data.shape[0]] * generated_data.shape[0], dtype=torch.float32)
    # real_distribution = torch.tensor([1 / real_data.shape[0]] * real_data.shape[0], dtype=torch.float32)
    generated_distribution = torch.tensor(np.ones(generated_data.shape[0]) / generated_data.shape[0], dtype=torch.float32)
    real_distribution = torch.tensor(np.ones(real_data.shape[0]) / real_data.shape[0], dtype=torch.float32)

    print('Generated distribution: ', generated_distribution.shape)
    print('Real distribution: ', real_distribution.shape)

    wasserstein_distance = ot.emd2(generated_distribution, real_distribution, cost_matrix)

    return wasserstein_distance

def log(msg: str, file_path: str, to_console: bool = False):
    if to_console:
        print(msg)
    with open(file_path, 'a') as f:
        f.write(msg + '\n')

def main(args):    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'

    # Load data.
    x, _, labels, _ = load_data(args.data_path)
    # Split into train/val/test.
    train_idx, val_idx, test_idx = split_train_val_test(x, test_size=args.test_size, val_size=args.val_size)
    train_x, train_labels = x[train_idx], labels[train_idx]
    # val_x, val_labels = x[val_idx], labels[val_idx]
    test_x, test_labels = x[test_idx], labels[test_idx]
    print(f'[Train] train_x: {train_x.shape}, train_labels: {train_labels.shape}')
    # print(f'[Val] val_x: {val_x.shape}, val_labels: {val_labels.shape}')
    print(f'[Test] test_x: {test_x.shape}, test_labels: {test_labels.shape}')

    # Load models.
    if args.train_autoencoder:
        raise NotImplementedError('Training new autoencoder.')
    elif args.use_local_ae:
        print('Loading local autoencoder...')
        ae_model = load_local_autoencoder(f'{args.ae_checkpoint_dir}/{args.autoencoder_ckptname}')
    elif args.ae_use_pretrained:
        print('Loading pretrained Unified autoencoder...')
        ae_model = load_unified_autoencoder(f'{args.checkpoint_dir}/{args.autoencoder_ckptname}')
    else:
        print('Loading autoencoder from wandb...')
        ae_model = load_autoencoder(args.ae_run_id, args.root_dir)

    print(f'[Data Loaded] Unique labels: {np.unique(labels)}')

    ''' Train new discriminator. '''
    if args.disc_use_gp:
        raise NotImplementedError('GP not implemented.')
    elif args.disc_use_pretrained:
        print('Loading pretrained discriminator...')
        wd_model = Discriminator.load_from_checkpoint(f'{args.checkpoint_dir}/{args.discriminator_ckptname}')
    else:
        raise NotImplementedError('Training new discriminator.')

    ''' ====== Train GeodesicFM ====== '''
    # empty cache
    torch.cuda.empty_cache()
    torch.mps.empty_cache()

    device = 'cpu'
    if args.use_all_group_points: # Use all points in the start/end group.
        train_start_pts = train_x[train_labels == args.start_group]
        train_end_pts = train_x[train_labels == args.end_group]
        test_start_pts = test_x[test_labels == args.start_group]
        test_end_pts = test_x[test_labels == args.end_group]
    else: # Sample start/end points in start/end groups.
        raise NotImplementedError('Sampling start/end points.')

    # Create dataloader.
    print('[Training] Start/End Points: ', train_start_pts.shape, train_end_pts.shape)

    train_dataset = CustomDataset(x0=torch.tensor(train_start_pts, dtype=torch.float32), 
                                  x1=torch.tensor(train_end_pts, dtype=torch.float32))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Prepare offmanifolder through encoder and discriminator.
    ae_model = ae_model.to(device)
    wd_model = wd_model.to(device)
    ae_model.eval()
    wd_model.eval()
    for param in ae_model.encoder.parameters():
        param.requires_grad = False
    for param in wd_model.parameters():
        param.requires_grad = False

    enc_func = lambda x: ae_model.encoder(x)
    alpha = args.alpha
    disc_func = lambda x: torch.exp(alpha * (1 - (wd_model.positive_prob(enc_func(x)))))
    # distance_func_in_latent = lambda z: torch.exp(alpha * (1 - wd_model.positive_prob(z)))
    ofm = offmanifolder_maker_new(enc_func, disc_func, 
                                  disc_factor=args.disc_factor, 
                                  data_encodings=None)
    
    gbmodel = GeodesicFM(
        func=ofm,
        encoder=enc_func,
        input_dim=x.shape[1],
        hidden_dim=args.hidden_dim, 
        scale_factor=args.scale_factor, 
        symmetric=args.symmetric, 
        embed_t=args.embed_t,
        num_layers=args.num_layers, 
        n_tsteps=args.n_tsteps, 
        lr=args.lr,
        weight_decay=args.weight_decay,
        flow_weight=args.flow_weight,
        length_weight=args.length_weight,
        cc_k=args.cc_k,
        init_method=args.init_method,
        # data_pts_encodings=torch.tensor(x_encodings, dtype=torch.float32).to(device),
        use_density=args.use_density,
        # data_pts=torch.tensor(x, dtype=torch.float32).to(device),
        # diff_op=diff_op,
        # diff_t=diff_t,
        density_weight=args.density_weight,
        fixed_pot=args.fixed_pot,
        visualize_training=args.visualize_training,
        dataloader=train_dataloader, # for visualization, not used for training.
        device=device,
        training_save_dir=args.training_save_dir,
    )

    # Train the model
    # TODO: use validation loss to early stop.
    early_stopping = pl.callbacks.EarlyStopping(monitor='train_loss_epoch', patience=args.patience, mode='min')
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor='train_loss_epoch', save_top_k=1, mode='min', 
                                                    dirpath=args.checkpoint_dir, filename='gbmodel')
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        accelerator=device,
        callbacks=[early_stopping, model_checkpoint]
    )

    trainer.fit(gbmodel, train_dataloaders=train_dataloader)
    #trainer.fit(gbmodel, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    ''' Visualization and Evaluation '''
    print('=========== Visualizing GeodesicBridge Paths ==========')
    log('[Visualization] Visualizing GeodesicBridge Paths ...', os.path.join(args.plots_save_dir, 'eval.log'))
    # NOTE: use test set here.
    start_pts = test_start_pts
    end_pts = test_end_pts
    
    n_samples = start_pts.shape[0]
    start_pts = start_pts[:n_samples]
    end_pts = end_pts[:n_samples]

    # Use Neural ODE to integrate the learned flow/vector field.
    print('=========== Running ODE on learned vector field ==========')
    n_tsteps = 100
    t_start = 0
    t_end = 1

    flow_ode = ODEFuncWrapper(gbmodel.flow_model).to('cpu')
    with torch.no_grad():
        ts = torch.linspace(t_start, t_end, n_tsteps).to('cpu')
        traj = odeint(flow_ode, torch.tensor(start_pts, dtype=torch.float32).to('cpu'), ts)
        
    print('ODE Trajectory shape: ', traj.shape)
    #encoded_traj = encode_data(traj.flatten(0,1), ae_model.encoder, 'cpu').reshape(n_tsteps, -1, x_encodings.shape[1]) # [n_tsteps, n_samples, latent_dim]
    #print('Encoded Trajectory shape: ', encoded_traj.shape)

    ''' Wasserstein 1 distance between generated and real data. '''
    print('====== Evaluating Wasserstein 1 distance between generated and gt ======')
    test_group = args.test_group
    # TODO: which samples to use for real data/generated data.
    real_idx = np.where(test_labels == test_group)[0]
    real_data = test_x[real_idx]
    generated_data = traj.flatten(0,1) # [n_tsteps*n_samples, ambient_dim]
    print('[Eval] Real data shape: ', real_data.shape, 'Generated data shape: ', generated_data.shape)
    # TODO: do we need to have the same number of samples for real and generated data?
    # num_samples = min(real_idx.shape[0], generated_data.shape[0])
    # generated_data_idx = np.random.choice(np.arange(generated_data.shape[0]), size=num_samples, replace=False)
    # generated_data = generated_data[generated_data_idx]
    # real_samples_idx = np.random.choice(np.arange(real_data.shape[0]), size=num_samples, replace=False)
    # real_data = real_data[real_samples_idx]
    
    wasserstein_distance = eval_distributions(generated_data, real_data, cost_metric='euclidean')
    print('[Eval] Wasserstein-1 distance: ', wasserstein_distance.item())
    log(f'[Eval] Wasserstein-1 distance with target test group {test_group}: {wasserstein_distance.item()}', os.path.join(args.plots_save_dir, 'eval.log'))

    print('Evaluation finished.')


def eval(args):
    """
        Evaluate the model on the manifold data.
    NOTE: need to load correct 1) autoencoder, 2) discriminator, and 3) gbmodel, 4) dataset.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'

    # Load models
    if args.train_autoencoder:
        print('Loading Unified-Trained autoencoder...')
        ae_model = load_unified_autoencoder(f'{args.checkpoint_dir}/{args.autoencoder_ckptname}')
    elif args.use_local_ae:
        print('Loading local autoencoder...')
        #ae_model = load_local_autoencoder(args.ae_checkpoint_path)
        ae_model = load_local_autoencoder(f'{args.ae_checkpoint_dir}/{args.autoencoder_ckptname}')
    else:
        print('Loading autoencoder from wandb...')
        ae_model = load_autoencoder(args.ae_run_id, args.root_dir)
    # wd_model = Discriminator.load_from_checkpoint(f'{args.checkpoint_dir}/discriminator-v3.ckpt')
    wd_model = Discriminator.load_from_checkpoint(f'{args.checkpoint_dir}/{args.discriminator_ckptname}')

    # Load data
    x, x_distances, labels, phate_coords = load_data(args.data_path)
    x_encodings = encode_data(x, ae_model.encoder, device)
    train_idx, val_idx, test_idx = split_train_val_test(x, test_size=args.test_size, val_size=args.val_size)
    train_x, train_x_encodings, train_labels = x[train_idx], x_encodings[train_idx], labels[train_idx]
    val_x, val_x_encodings, val_labels = x[val_idx], x_encodings[val_idx], labels[val_idx]
    test_x, test_x_encodings, test_labels = x[test_idx], x_encodings[test_idx], labels[test_idx]

    # Load gbmodel.
    ae_model = ae_model.to(device)
    wd_model = wd_model.to(device)
    ae_model.eval()
    wd_model.eval()
    for param in ae_model.encoder.parameters():
        param.requires_grad = False
    for param in wd_model.parameters():
        param.requires_grad = False

    enc_func = lambda x: ae_model.encoder(x)
    alpha = args.alpha
    disc_func = lambda x: torch.exp(alpha * (1 - (wd_model.positive_prob(enc_func(x)))))
    distance_func_in_latent = lambda z: torch.exp(alpha * (1 - wd_model.positive_prob(z)))
    ofm = offmanifolder_maker_new(enc_func, disc_func, 
                                  disc_factor=args.disc_factor, 
                                  data_encodings=torch.tensor(x_encodings, dtype=torch.float32).to(device))
    
    ''' Select start, end points '''
    # Dummy, just to initialize the Geodesic Flow Matching model.
    if args.use_all_group_points: # Use all points in the start/end group.
        train_start_pts = train_x[train_labels == args.start_group]
        train_end_pts = train_x[train_labels == args.end_group]
        val_start_pts = val_x[val_labels == args.start_group]
        val_end_pts = val_x[val_labels == args.end_group]
        test_start_pts = test_x[test_labels == args.start_group]
        test_end_pts = test_x[test_labels == args.end_group]
    else: # Sample start/end points from within a range.
        train_start_idx, train_sampled_indices_point1, train_end_idx, train_sampled_indices_point2 = sample_indices_within_range(
            x=train_x, encoder=ae_model.encoder, device=device, labels=train_labels, 
            start_group=args.start_group, end_group=args.end_group, 
            selected_idx=(args.start_idx, args.end_idx), range_size=args.range_size, num_samples=args.num_samples, 
            seed=args.seed, 
        )
        train_start_pts = train_x[train_sampled_indices_point1]
        train_end_pts = train_x[train_sampled_indices_point2]
        val_start_idx, val_sampled_indices_point1, val_end_idx, val_sampled_indices_point2 = sample_indices_within_range(
            x=val_x, encoder=ae_model.encoder, device=device, labels=val_labels, 
            start_group=args.start_group, end_group=args.end_group, 
            selected_idx=(args.start_idx, args.end_idx), range_size=args.range_size, num_samples=args.num_samples, 
            seed=args.seed, 
        )
        val_start_pts = val_x[val_sampled_indices_point1]
        val_end_pts = val_x[val_sampled_indices_point2]
        test_start_idx, test_sampled_indices_point1, test_end_idx, test_sampled_indices_point2 = sample_indices_within_range(
            x=test_x, encoder=ae_model.encoder, device=device, labels=test_labels, 
            start_group=args.start_group, end_group=args.end_group, 
            selected_idx=(args.start_idx, args.end_idx), range_size=args.range_size, num_samples=args.num_samples, 
            seed=args.seed, 
        )
        test_start_pts = test_x[test_sampled_indices_point1]
        test_end_pts = test_x[test_sampled_indices_point2]

    # Split into train/val/test.
    print('[Total] All Points: ', x.shape)
    print('[Train] Start/End Points: ', train_start_pts.shape, train_end_pts.shape)
    print('[Val] Start/End Points: ', val_start_pts.shape, val_end_pts.shape)
    print('[Test] Start/End Points: ', test_start_pts.shape, test_end_pts.shape)

    train_dataset = CustomDataset(x0=torch.tensor(train_start_pts, dtype=torch.float32), 
                                 x1=torch.tensor(train_end_pts, dtype=torch.float32))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataset = CustomDataset(x0=torch.tensor(val_start_pts, dtype=torch.float32), 
                                x1=torch.tensor(val_end_pts, dtype=torch.float32))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_dataset = CustomDataset(x0=torch.tensor(test_start_pts, dtype=torch.float32), 
                                 x1=torch.tensor(test_end_pts, dtype=torch.float32))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    diff_op = None
    diff_t = 3
    if args.init_method == 'diffusion':
        # Get diff_op from phate op
        phate_op = phate.PHATE(n_components=x_encodings.shape[1], 
                               n_landmark=x_encodings.shape[0],
                               knn=10,
                               verbose=True).fit(x)
        diff_op = phate_op.diff_op
        print('[PHATE] diff_op: ', diff_op.shape)
    gbmodel = GeodesicFM.load_from_checkpoint(
        f'{args.checkpoint_dir}/{args.gbmodel_ckptname}', 
        func=ofm,
        encoder=enc_func,
        input_dim=x.shape[1],
        hidden_dim=args.hidden_dim, 
        scale_factor=args.scale_factor, 
        symmetric=args.symmetric, 
        embed_t=args.embed_t,
        num_layers=args.num_layers, 
        n_tsteps=args.n_tsteps, 
        lr=args.lr,
        weight_decay=args.weight_decay,
        flow_weight=args.flow_weight,
        length_weight=args.length_weight,
        cc_k=args.cc_k,
        init_method=args.init_method,
        #data_pts_encodings=torch.tensor(x_encodings, dtype=torch.float32).to(device),
        use_density=args.use_density,
        #data_pts=torch.tensor(x, dtype=torch.float32).to(device),
        diff_op=diff_op,
        diff_t=diff_t,
        density_weight=args.density_weight,
        fixed_pot=args.fixed_pot,
        visualize_training=args.visualize_training,
        dataloader=train_dataloader,
        device=device,
        training_save_dir=args.training_save_dir,
    )

    ''' Visualize the learned Geodesic Paths. '''
    print('=========== Visualizing GeodesicBridge Paths ==========')
    log('[Visualization] Visualizing GeodesicBridge Paths ...', os.path.join(args.plots_save_dir, 'eval.log'))

    # TODO: this start/ends is ALL pts, may want to use test set here.
    # NOTE: use test set here.
    start_pts = test_start_pts
    end_pts = test_start_pts

    n_samples = min(start_pts.shape[0], end_pts.shape[0])
    #n_samples = 100
    n_samples = start_pts.shape[0]

    start_pts = start_pts[:n_samples]
    end_pts = end_pts[:n_samples]
    print('[Eval] Start/End Points: ', start_pts.shape, end_pts.shape)
    log(f'[Eval] Start/End Points: {start_pts.shape}, {end_pts.shape}', os.path.join(args.plots_save_dir, 'eval.log'))
    dummy_ids = torch.zeros((start_pts.shape[0], 1), dtype=torch.float32)

    gbmodel.to(device)
    n_tsteps = args.n_tsteps
    n_tsteps = 100
    t_start = 0
    t_end = 1.0
    gb_trajs = gbmodel.cc(torch.tensor(start_pts, dtype=torch.float32).to(device), 
                           torch.tensor(end_pts, dtype=torch.float32).to(device), 
                           torch.tensor(np.linspace(t_start, t_end, n_tsteps), dtype=torch.float32).to(device),
                           dummy_ids.to(device))  
         
    start_pts_encodings = encode_data(start_pts, ae_model.encoder, device)
    end_pts_encodings = encode_data(end_pts, ae_model.encoder, device)                
    gb_trajs_encodings = encode_data(gb_trajs.flatten(0,1), ae_model.encoder, device).reshape(n_tsteps, -1, x_encodings.shape[1]) # [n_tsteps, n_samples, latent_dim]

    visualize_trajectory(gb_trajs_encodings, x_encodings, labels,
                         save_path=os.path.join(args.plots_save_dir, f'eval_geodesic_paths_latent.png'), 
                         title=f'[eval] Geodesic Flow Paths in Latent Space', 
                         start_pts=start_pts_encodings, end_pts=end_pts_encodings, plotly=args.plotly)


    ''' Neural ODE to integrate the learned flow/vector field. '''
    print('=========== Running ODE on learned vector field ==========')
    # n_samples = min(100, start_pts.shape[0])
    flow_ode = ODEFuncWrapper(gbmodel.flow_model).to('cpu')
    log(f'[Eval] Running ODE on learned vector field ...', os.path.join(args.plots_save_dir, 'eval.log'))
    log(f'[Eval] Start Group: {args.start_group}, End Group: {args.end_group}, Test Group: {args.test_group}', os.path.join(args.plots_save_dir, 'eval.log'))
    log(f'[Eval] Number of time steps: {n_tsteps}', os.path.join(args.plots_save_dir, 'eval.log'))
    log(f'[Eval] Start time: {t_start}', os.path.join(args.plots_save_dir, 'eval.log'))
    log(f'[Eval] End time: {t_end}', os.path.join(args.plots_save_dir, 'eval.log'))
    with torch.no_grad():
        ts = torch.linspace(0, 1, n_tsteps).to('cpu')
        # traj = odeint(flow_ode, sampled_starts, ts) # [n_tsteps, n_samples, ambient_dim]
        traj = odeint(flow_ode, torch.tensor(start_pts, dtype=torch.float32).to('cpu'), ts)
        
    print('Flow Matching ODE Trajectory shape: ', traj.shape)
    encoded_traj = encode_data(traj.flatten(0,1), ae_model.encoder, 'cpu').reshape(n_tsteps, -1, start_pts_encodings.shape[1]) # [n_tsteps, n_samples, latent_dim]
    print('Encoded Trajectory shape: ', encoded_traj.shape)

    # Visualize the ODE trajectory in latent space.
    visualize_trajectory(encoded_traj, x_encodings, labels,
                         save_path=os.path.join(args.plots_save_dir, f'eval_ode_latent_traj.png'), 
                         title=f'[eval] ODE Trajectory in Latent Space', 
                         start_pts=start_pts_encodings, end_pts=end_pts_encodings, plotly=args.plotly)

    ''' Wasserstein 1 distance between generated and real data. '''
    print('====== Evaluating Wasserstein 1 distance between generated and gt ======')
    test_group = args.test_group
    # TODO: which samples to use for real data/generated data.
    real_idx = np.where(test_labels == test_group)[0]
    real_data = test_x[real_idx]
    # real_idx = np.where(labels != test_group)[0]
    # real_data = x[real_idx]
    #traj = traj[5:-5, : , :]
    # sampled_ts = [int(n_tsteps * 0.2), int(n_tsteps * 0.9)]
    # traj = traj[sampled_ts[0]:sampled_ts[1], :, :] # [sampled_n_tsteps, n_samples, ambient_dim]
    generated_data = traj.flatten(0,1) # [sampled_n_tsteps*n_samples, ambient_dim]

    # TODO: do we need to have the same number of samples for real and generated data?
    # num_samples = min(real_idx.shape[0], generated_data.shape[0])
    # generated_data_idx = np.random.choice(np.arange(generated_data.shape[0]), size=num_samples, replace=False)
    # generated_data = generated_data[generated_data_idx]
    # real_samples_idx = np.random.choice(np.arange(real_data.shape[0]), size=num_samples, replace=False)
    # real_data = real_data[real_samples_idx]
    print('[Eval] Real data shape: ', real_data.shape, 'Generated data shape: ', generated_data.shape)
    log(f'[Eval] Real data shape: {real_data.shape}, Generated data shape: {generated_data.shape}', os.path.join(args.plots_save_dir, 'eval.log'))
    
    wasserstein_distance = eval_distributions(generated_data, real_data, cost_metric='euclidean')
    print('[Eval] Wasserstein-1 distance: ', wasserstein_distance.item())
    log(f'[Eval] W1 distance with target group {test_group}: {wasserstein_distance.item()}', os.path.join(args.plots_save_dir, 'eval.log'))
    # Plot the generated and real data in latent space.
    real_data_encodings = test_x_encodings[real_idx]
    generated_data_encodings = encoded_traj # [n_tsteps*n_samples, latent_dim]
    visualize_generated(generated_data_encodings, real_data_encodings, x_encodings, labels,
                        save_path=os.path.join(args.plots_save_dir, f'eval_generated_real_latent_t{test_group}.png'), 
                        title=f'[eval] Generated and Real Data at t={test_group} in Latent Space', 
                        plotly=args.plotly)


    log('[Eval] Evaluation finished.\n', os.path.join(args.plots_save_dir, 'eval.log'), to_console=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Discriminator Script")
    parser.add_argument("--mode", type=str, default='train', help="train|eval")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--root_dir", type=str, default="../../", help="Root directory")
    parser.add_argument("--use_local_ae", action='store_true', help="Use locally trained autoencoder")
    parser.add_argument("--ae_checkpoint_dir", type=str, default='./ae_checkpoints', help="Path to the (locally) trained autoencoder checkpoint")
    parser.add_argument("--ae_run_id", type=str, default='pzlwi6t6', help="Autoencoder run ID")
    parser.add_argument("--discriminator_ckptname", type=str, default='discriminator-v3.ckpt', help="Discriminator checkpoint name")
    parser.add_argument("--autoencoder_ckptname", type=str, default='autoencoder-v17.ckpt', help="Autoencoder checkpoint name")
    parser.add_argument("--gbmodel_ckptname", type=str, default='gbmodel-v3.ckpt', help="GeodesicFM checkpoint name")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--plots_save_dir", type=str, default="./plots", help="Save directory")
    parser.add_argument("--data_path", type=str, default='../../data/eb_subset_all.npz')
    # Start/End points arguments
    parser.add_argument("--plotly", action='store_true', help="Use plotly for visualization")
    parser.add_argument("--use_all_group_points", action='store_true', help="Use all group points in start/end group to train trajectory")
    parser.add_argument("--sample_group_points", action='store_true', help="Randomly sample start/end group points in start/end group for trajectory")
    parser.add_argument("--start_group", type=int, default=0, help="Start group for trajectory")
    parser.add_argument("--end_group", type=int, default=2, help="End group for trajectory")
    parser.add_argument("--start_idx", type=int, default=736, help="Start index for trajectory")
    parser.add_argument("--end_idx", type=int, default=2543, help="End index for trajectory")
    parser.add_argument("--range_size", type=float, default=0.3, help="Range size for sampling")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--test_group", type=int, default=1, help="Test group for evaluation")
    # GeodesicFM arguments
    parser.add_argument("--test_size", type=float, default=0.1, help="Test size")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation size")
    parser.add_argument("--alpha", type=float, default=10, help="Alpha for off-manifolder proxy")
    parser.add_argument("--disc_factor", type=float, default=5, help="Discriminator factor for off-manifolder")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for GeodesicFM")
    parser.add_argument("--scale_factor", type=float, default=1, help="Scale factor for CondCurve")
    parser.add_argument("--symmetric", action='store_true', help="Symmetric flag for GeodesicFM")
    parser.add_argument("--fixed_pot", action='store_true', help="Fixed x1,x0 pairs for GeodesicFM")
    parser.add_argument("--embed_t", action='store_true', help="Embed t for GeodesicFM")
    parser.add_argument("--init_method", type=str, default='line', help="Init method for GeodesicFM")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers for GeodesicFM")
    parser.add_argument("--n_tsteps", type=int, default=100, help="Number of time steps for GeodesicFM")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--flow_weight", type=float, default=1, help="Flow weight for GeodesicFM")
    parser.add_argument("--length_weight", type=float, default=1, help="Length weight for GeodesicFM")
    parser.add_argument("--cc_k", type=int, default=2, help="cc_k parameter for GeodesicFM CondCurve")
    parser.add_argument("--use_density", action='store_true', help="Use density for GeodesicFM")
    parser.add_argument("--density_weight", type=float, default=1., help="Density weight for GeodesicFM")
    parser.add_argument("--visualize_training", action='store_true', help="Visualize training")
    parser.add_argument("--training_save_dir", type=str, default="./eb_fm/training/", help="Save directory")
    parser.add_argument("--patience", type=int, default=150, help="Patience for early stopping")
    parser.add_argument("--max_epochs", type=int, default=300, help="Maximum number of epochs")
    parser.add_argument("--log_every_n_steps", type=int, default=20, help="Log every n steps")
    parser.add_argument("--show_plot", action='store_true', help="Show plot")
    # Autoencoder training
    parser.add_argument("--ae_use_pretrained", action='store_true', help="Use pretrained autoencoder")
    parser.add_argument("--train_autoencoder", action='store_true', help="Train autoencoder")
    parser.add_argument("--ae_component_wise_normalization", action='store_true', help="Use component-wise normalization for autoencoder")
    parser.add_argument("--ae_batch_size", type=int, default=256, help="Batch size for autoencoder")
    parser.add_argument("--ae_max_epochs", type=int, default=200, help="Maximum number of epochs for training")
    parser.add_argument("--ae_log_every_n_steps", type=int, default=100, help="Log every n steps for autoencoder")
    parser.add_argument("--ae_early_stop_patience", type=int, default=50, help="Early stop patience for autoencoder")
    parser.add_argument("--ae_latent_dim", type=int, default=3, help="Latent dimension")
    parser.add_argument("--ae_batch_norm", action='store_true', help="Use batch normalization for autoencoder")
    parser.add_argument("--ae_dropout", type=float, default=0.2, help="Dropout rate for autoencoder")
    parser.add_argument("--ae_use_spectral_norm", action='store_true', help="Use spectral normalization for autoencoder")
    parser.add_argument("--ae_dist_mse_decay", type=float, default=0.0, help="Decay rate for distance loss")
    parser.add_argument("--ae_weights_dist", type=float, default=77.4, help="Weight for distance loss")
    parser.add_argument("--ae_weights_reconstr", type=float, default=0.32, help="Weight for reconstruction loss")
    parser.add_argument("--ae_weights_cycle", type=float, default=1, help="Weight for cycle loss")
    parser.add_argument("--ae_weights_cycle_dist", type=float, default=0, help="Weight for cycle distance loss")
    parser.add_argument("--ae_lr", type=float, default=1e-3, help="Learning rate for autoencoder")
    parser.add_argument("--ae_weight_decay", type=float, default=1e-4, help="Weight decay for autoencoder")
    parser.add_argument("--ambient_source", type=str, default='pca', help="Ambient source")
    parser.add_argument("--ae_encoder_layer_width", type=int, nargs="+", default=[256, 128, 64], help="Encoder layer widths for autoencoder")
    parser.add_argument("--ae_decoder_layer_width", type=int, nargs="+", default=[64, 128, 256], help="Decoder layer widths for autoencoder")
    parser.add_argument("--ae_activation", type=str, default='relu', help="Activation function for autoencoder")

    # Add new arguments for discriminator training
    parser.add_argument("--disc_use_pretrained", action='store_true', help="Use pretrained discriminator")
    parser.add_argument("--disc_layer_widths", type=int, nargs="+", default=[256, 128, 64], help="Layer widths for discriminator")
    parser.add_argument("--disc_lr", type=float, default=1e-3, help="Learning rate for discriminator")
    parser.add_argument("--disc_weight_decay", type=float, default=1e-4, help="Weight decay for discriminator")
    parser.add_argument("--disc_early_stop_patience", type=int, default=50, help="Early stop patience for discriminator")
    parser.add_argument("--disc_max_epochs", type=int, default=100, help="Number of epochs for discriminator training")
    parser.add_argument("--disc_log_every_n_steps", type=int, default=20, help="Log every n steps for discriminator")
    parser.add_argument("--disc_batch_size", type=int, default=64, help="Batch size for discriminator training")
    parser.add_argument("--disc_use_density_features", action='store_true', help="Use density features for discriminator")
    parser.add_argument("--disc_density_k", type=int, default=5, help="k for density features")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    # GP arguments
    parser.add_argument("--disc_use_gp", action='store_true', help="Use GP for discriminator")
    # Negative sampling arguments    
    parser.add_argument("--neg_method", type=str, default="add", help="add|diffusion")
    parser.add_argument("--noise_rate", type=float, default=1.0, help="std of the additive noise")
    parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.2], help="Noise levels for negative sampling")
    parser.add_argument("--noise_type", type=str, default="gaussian", help="Type of noise for negative sampling")
    parser.add_argument("--t", type=int, nargs="+", default=[200], help="Number of time steps for forward diffusion")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps for forward diffusion")
    parser.add_argument("--beta_start", type=float, default=0.0001, help="Start value for beta in forward diffusion")
    parser.add_argument("--beta_end", type=float, default=0.01, help="End value for beta in forward diffusion")
    parser.add_argument("--sampling_rejection", action='store_true', help="Sampling rejection for negative sampling")
    parser.add_argument("--sampling_rejection_method", type=str, default="density", help="density|sugar")
    parser.add_argument("--sampling_rejection_k", type=int, default=20, help="k for sampling rejection")
    parser.add_argument("--sampling_rejection_threshold", type=float, default=.5, help="Threshold for sampling rejection")


    args = parser.parse_args()

    data_filename = os.path.basename(args.data_path)
    data_name = data_filename.split('.')[0]
    args.plots_save_dir = f"{data_name}_t-{args.test_group}_alpha-{args.alpha}_factor-{args.disc_factor}_{os.path.basename(args.plots_save_dir)}"
    args.checkpoint_dir = f"{data_name}_t-{args.test_group}_alpha-{args.alpha}_factor-{args.disc_factor}_{os.path.basename(args.checkpoint_dir)}"
    args.ae_checkpoint_dir = f"{data_name}_t-{args.test_group}_{os.path.basename(args.ae_checkpoint_dir)}"
    print('args.plots_save_dir: ', args.plots_save_dir)
    print('args.checkpoint_dir: ', args.checkpoint_dir)
    print('args.ae_checkpoint_dir: ', args.ae_checkpoint_dir)

    os.makedirs(args.plots_save_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print('args: ', args)
    if args.mode == 'train':
        main(args)
    elif args.mode == 'eval':
        eval(args)

