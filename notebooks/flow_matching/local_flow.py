import argparse
import os
import sys
from glob import glob
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torchdiffeq
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import gpytorch
from gpytorch.kernels import RBFKernel, MaternKernel, PolynomialKernel, LinearKernel, ScaleKernel, AdditiveKernel
import scipy.spatial as spatial

sys.path.append('../../src/')
from diffusionmap import DiffusionMap
from negative_sampling import make_hi_freq_noise
from model2 import Autoencoder, Discriminator as OldDiscriminator
from off_manifolder import offmanifolder_maker_new
from geodesic import GeodesicFM


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

# Define a simple GP model
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def predict_uncertainty(self, x):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self(x)
        return posterior.variance
    
    def positive_prob(self, x):
        '''to be compatible with other discriminators'''
        return -self.predict_uncertainty(x)

def init_gp(x, y) -> GPModel:
    '''
    x: (n_samples, latent_dim)
    y: (n_samples, )
    Compute the GP posterior for test points conditioned on (x, y).
    '''
    #import pdb; pdb.set_trace()

    # Assert x and y are torch tensors.
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    # Fit GP model.
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    model = GPModel(x, y, likelihood)

    # Switch to evaluation mode for posterior computation
    model.eval()
    likelihood.eval()

    return model

def train_discriminator(x, x_noisy, encoder, args):
    # Dataloader.
    X = torch.cat([x, x_noisy], dim=0)
    Y = torch.cat([torch.ones(x.shape[0]), torch.zeros(x_noisy.shape[0])], dim=0)
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
                         callbacks=[early_stop, model_checkpoint])
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Test model.
    trainer.test(ckpt_path="best", dataloaders=test_dataloader)

    # Load best model.
    best_model = Discriminator.load_from_checkpoint(model_checkpoint.best_model_path)
    
    return best_model
    
def load_autoencoder(run_id, root_dir):
    run_path = os.path.join(root_dir, 'src/wandb/')
    run_path = glob(f"{run_path}/*{run_id}")[0]
    model_path = glob(f"{run_path}/files/*.ckpt")[0]
    return Autoencoder.load_from_checkpoint(model_path)

def load_discriminator(run_id, root_dir):
    run_path = os.path.join(root_dir, 'src/wandb/')
    run_path = glob(f"{run_path}/*{run_id}")[0]
    model_path = glob(f"{run_path}/files/*.ckpt")[0]
    return OldDiscriminator.load_from_checkpoint(model_path)

def load_data(data_path):
    data = np.load(data_path)
    return data['data'], data['colors']

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

    # Randomly select two points from the array
    if selected_idx[0] is None or selected_idx[1] is None:
        assert labels is not None and start_group is not None and end_group is not None
        start_idxs = np.where(labels == start_group)[0]
        end_idxs = np.where(labels == end_group)[0]
        point1_idx, point2_idx = np.random.choice(start_idxs, 1)[0], np.random.choice(end_idxs, 1)[0]
        point1, point2 = points[point1_idx], points[point2_idx]
    else:
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
        t: (1, )
        x: (n_samples, n_features)
        '''
        # Expand t to match the batch size and feature dimension
        t_expanded = t.view(1, 1).expand(x.size(0), 1)  # (n_samples, 1)
        # Concatenate x and t along the feature dimension
        x_with_t = torch.cat((x, t_expanded), dim=-1) # (n_samples, n_features + 1)
        return self.flow_model(x_with_t) # (n_samples, n_features)
    
class FlowMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(FlowMLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class FlowMatching(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers, local_flow_models, local_n_tsteps=100,
                 lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.flow_model = FlowMLP(input_dim=input_dim+1, hidden_dim=hidden_dim, output_dim=input_dim, num_hidden_layers=num_layers)
        self.local_flow_models = local_flow_models # list of flow models, each for a local region.
        self.local_n_tsteps = local_n_tsteps
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x_with_t):
        '''
        x_with_t: (n_samples, n_features + 1)
        return: vector field (n_samples, n_features) at x at time t.
        '''

        return self.flow_model(x_with_t) # (n_samples, n_features)
    
    def training_step(self, batch):
        batch_x_t, batch_t, batch_t_idx = batch # x_t: (batch_size, n_samples, n_features), t: (batch_size, 1), t_idx: (batch_size, 1)
        #print('training step...', batch_x_t.shape, batch_t, batch_t_idx)
        segment_idx = list(torch.tensor(batch_t_idx // self.local_n_tsteps, dtype=torch.int8).cpu().numpy())
        #print('segment_idx: ', segment_idx)
        gt_flow_models = [(self.local_flow_models[i], i) for i in segment_idx]
        loss = 0
        for i, (gt_flow_model, seg_idx) in enumerate(gt_flow_models):
            local_flow_model = gt_flow_model.flow_model.to(self.device)
            x_t = batch_x_t[i]
            t = batch_t[i].view(1,)
            global_t_expanded = t.unsqueeze(1).expand(x_t.shape[0], 1) # (n_samples, 1)
            local_t = t - seg_idx * 1 # range [0, 1]
            #print('global t: ', t, 'local t: ', local_t)
            local_t_expanded = local_t.unsqueeze(1).expand(x_t.shape[0], 1) # (n_samples, 1)
            #print('local_t_expanded: ', local_t_expanded.shape)
            # global_t_expanded = ((1/len(self.local_flow_models))*seg_idx+t/len(self.local_flow_models)) * torch.ones_like(local_t_expanded) # (n_samples, 1)
            #print('global_t_expanded: ', global_t_expanded.shape)

            pred_vec = self.forward(torch.cat([x_t, global_t_expanded], dim=1))
            gt_vec = local_flow_model(torch.cat([x_t, local_t_expanded], dim=1))
            loss += ((pred_vec - gt_vec) ** 2).mean() * (torch.exp(t - 1)**2) 
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

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

class FlowMatchingDataset():
    def __init__(self, global_trajs, n_tsteps, n_local_flow_models):
        '''
        global_trajs: (n_tsteps * n_local_flow_models, n_samples, n_features)
        n_tsteps: int, number of time steps for the each local flow.
        n_local_flow_models: int, number of local flow models.
        getitem:
            (x_t, t, t_idx): x_t: (batch_size, n_samples, n_features), t: (batch_size, 1), t_idx: (batch_size, 1)
        '''
        self.global_trajs = global_trajs
        self.n_tsteps = n_tsteps
        self.n_local_flow_models = n_local_flow_models

        assert global_trajs.shape[0] == n_local_flow_models * n_tsteps
    
    def __len__(self):
        return self.n_tsteps * self.n_local_flow_models - 1 # skip the mid point to avoid local sink problem.
    
    def __getitem__(self, idx):
        t_idx = idx
        t = torch.tensor(t_idx / (self.n_local_flow_models * self.n_tsteps) * self.n_local_flow_models, dtype=torch.float32) # range [0, 1*n_local_flow_models]
        x_t = torch.tensor(self.global_trajs[t_idx], dtype=torch.float32)
        return x_t, t, t_idx

def main(args):    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'

    # Load models
    ae_model = load_autoencoder(args.ae_run_id, args.root_dir)

    # Load data
    x, labels = load_data(args.data_path)
    x_encodings = encode_data(x, ae_model.encoder, device)
    print('x_encodings: ', x_encodings.shape)
    # Merge label 1, 2, 3, 4 into 1
    labels = np.where(labels == 2, 1, labels)
    labels = np.where(labels == 3, 1, labels)
    labels = np.where(labels == 4, 1, labels)
    print('Unique labels: ', np.unique(labels))


    os.makedirs(args.plots_save_dir, exist_ok=True)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for (i, label) in enumerate(np.unique(labels)):
        idx = np.where(labels == label)[0]
        ax.scatter(x_encodings[idx,0], x_encodings[idx,1], x_encodings[idx,2], c=[i]*len(idx), cmap='viridis', alpha=0.8, s=1)
    ax.set_title('Latent space')
    plt.savefig(os.path.join(args.plots_save_dir, 'latent_space.png'))

    # Negative sampling    
    if args.neg_method == 'add':
        x_noisy = neg_sample_using_additive(x_encodings, args.noise_levels, noise_rate=args.noise_rate, seed=args.seed, noise=args.noise_type)
    elif args.neg_method == 'diffusion':
        x_noisy = neg_sample_using_diffusion(x_encodings, args.t, args.num_steps, args.beta_start, args.beta_end, seed=args.seed)
    else:
        raise ValueError(f"Invalid negative sampling method: {args.neg_method}")
    assert x_noisy.shape[-1] == x_encodings.shape[-1]
    # Sampling rejection.
    if args.sampling_rejection:
        rejected_idx = sampling_rejection(x_encodings, x_noisy, 
                                          method=args.sampling_rejection_method, k=args.sampling_rejection_k, threshold=args.sampling_rejection_threshold)
        all_x_noisy = x_noisy.copy()
        x_noisy = x_noisy[~rejected_idx]
        print('Number of samples after sampling rejection: ', len(x_noisy))

    # Visualize negative samples and positive samples.
    # fig = go.Figure()
    # fig.add_trace(go.Scatter3d(x=x_encodings[:,0], y=x_encodings[:,1], z=x_encodings[:,2], 
    #                            mode='markers', marker=dict(size=2, color='gray', colorscale='Viridis', opacity=0.8)))
    # fig.add_trace(go.Scatter3d(x=x_noisy[:,0], y=x_noisy[:,1], z=x_noisy[:,2],
    #                            mode='markers', marker=dict(size=2, color='red', colorscale='Viridis', opacity=0.8)))
    # if args.sampling_rejection:
    #     fig.add_trace(go.Scatter3d(x=all_x_noisy[rejected_idx,0], y=all_x_noisy[rejected_idx,1], z=all_x_noisy[rejected_idx,2],
    #                                mode='markers', marker=dict(size=2, color='green', colorscale='Viridis', opacity=0.8)))
    # if args.show_plot:
    #     fig.show()
    # draw in plt
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_encodings[:,0], x_encodings[:,1], x_encodings[:,2], c='gray', cmap='viridis', alpha=0.8)
    ax.scatter(x_noisy[:,0], x_noisy[:,1], x_noisy[:,2], c='red', cmap='viridis', alpha=0.6)
    if args.sampling_rejection:
        ax.scatter(all_x_noisy[rejected_idx,0], all_x_noisy[rejected_idx,1], all_x_noisy[rejected_idx,2], c='green', cmap='viridis', alpha=0.4)
    ax.set_title('Positive and negative samples, and rejected samples')
    plt.savefig(os.path.join(args.plots_save_dir, 'pos_neg_rejected.png'))
    # 2D
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=x_encodings[:,0], y=x_encodings[:,1], mode='markers', marker=dict(size=2, color='gray', colorscale='Viridis', opacity=0.8)))
    # fig.add_trace(go.Scatter(x=x_noisy[:,0], y=x_noisy[:,1], mode='markers', marker=dict(size=2, color='red', colorscale='Viridis', opacity=0.8)))
    # if args.sampling_rejection:
    #     fig.add_trace(go.Scatter(x=all_x_noisy[rejected_idx,0], y=all_x_noisy[rejected_idx,1], mode='markers', marker=dict(size=2, color='green', colorscale='Viridis', opacity=0.8)))
    # if args.show_plot:
    #     fig.show()
    #return 
    
    # Train new discriminator
    if args.disc_use_gp:
        print('Initializing new GP discriminator... (no training needed)')
        wd_model = init_gp(torch.tensor(x_encodings, dtype=torch.float32), torch.ones(len(x_encodings), dtype=torch.float32).to(device))
    else:
        print("Training new discriminator...")
        print('args.disc_use_function_space: ', args.disc_use_function_space)
        if args.disc_use_function_space == False:
            print('Training new discriminator using standard CE loss.')
            wd_model = train_discriminator(torch.tensor(x_encodings, dtype=torch.float32),
                                    torch.tensor(x_noisy, dtype=torch.float32),
                                    ae_model.encoder,
                                    args)
        else:
            raise ValueError(f"Invalid discriminator training method: {args.disc_use_function_space}")

    # Visualize discriminator positive probs prediction on positive and negative samples.
    # Uniform sample N points from {-r, r} ^ latent_dim.
    latent_dim = x_encodings.shape[1]
    r = args.disc_fs_uniform_range
    uniform_samples = np.random.uniform(-r, r, size=(1000, latent_dim))
    wd_model.eval()
    wd_model.to(device)
    with torch.no_grad():
        pos_probs = wd_model.positive_prob(torch.tensor(x_encodings, dtype=torch.float32).to(device)).cpu().detach().numpy()
        neg_probs = wd_model.positive_prob(torch.tensor(x_noisy, dtype=torch.float32).to(device)).cpu().detach().numpy()
        uniform_probs = wd_model.positive_prob(torch.tensor(uniform_samples, dtype=torch.float32).to(device)).cpu().detach().numpy()
        if hasattr(wd_model, 'classify'):
            labels_pred = wd_model.classify(torch.tensor(x, dtype=torch.float32).to(device)).cpu().detach().numpy()
    print('pos_probs: ', pos_probs.mean())
    print('neg_probs: ', neg_probs.mean())
    print('uniform_probs: ', uniform_probs.mean())
    # fig = go.Figure()
    # fig.add_trace(go.Scatter3d(x=x_encodings[:,0], y=x_encodings[:,1], z=x_encodings[:,2], 
    #                            mode='markers', marker=dict(size=2, color=pos_probs, colorscale='Viridis', opacity=0.8)))
    # fig.add_trace(go.Scatter3d(x=x_noisy[:,0], y=x_noisy[:,1], z=x_noisy[:,2],
    #                            mode='markers', marker=dict(size=2, color=neg_probs, colorscale='Viridis', opacity=0.8)))
    # fig.add_trace(go.Scatter3d(x=uniform_samples[:,0], y=uniform_samples[:,1], z=uniform_samples[:,2],
    #                            mode='markers', marker=dict(size=2, color=uniform_probs, colorscale='Viridis', opacity=0.8)))
    # if hasattr(wd_model, 'classify'):
    #     fig.add_trace(go.Scatter3d(x=x_encodings[:,0], y=x_encodings[:,1], z=x_encodings[:,2], 
    #                             mode='markers', marker=dict(size=2, color=labels_pred, colorscale='Viridis', opacity=0.8),
    #                             text=labels_pred))
    # fig.update_layout(title='Discriminator positive probabilities')
    # if args.show_plot:
    #     fig.show()
    # draw in plt
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_encodings[:,0], x_encodings[:,1], x_encodings[:,2], c=pos_probs, cmap='viridis', alpha=0.8)
    ax.scatter(x_noisy[:,0], x_noisy[:,1], x_noisy[:,2], c=neg_probs, cmap='viridis', alpha=0.6)
    ax.scatter(uniform_samples[:,0], uniform_samples[:,1], uniform_samples[:,2], c=uniform_probs, cmap='viridis', alpha=0.4)
    if hasattr(wd_model, 'classify'):
        ax.scatter(x_encodings[:,0], x_encodings[:,1], x_encodings[:,2], c=labels_pred, cmap='viridis', alpha=0.8)
    ax.set_title('Discriminator positive probabilities')
    plt.savefig(os.path.join(args.plots_save_dir, 'disc_pos_probs.png'))

    # Select start/end points
    segment1 = (args.start_idx, args.mid_idx)
    segment2 = (args.mid_idx, args.end_idx)
    segments_pts = []
    local_gbmodels = []
    for segment in [segment1, segment2]:
        seg_start_idx, seg_end_idx = segment
        start_idx, sampled_indices_point1, end_idx, sampled_indices_point2 = sample_indices_within_range(
            x=x, encoder=ae_model.encoder, device=device, labels=labels, start_group=args.start_group, end_group=args.end_group, 
            selected_idx=(seg_start_idx, seg_end_idx), range_size=args.range_size, num_samples=args.num_samples, 
            seed=args.seed, 
        )
        start_pt = x[start_idx]
        end_pt = x[end_idx]
        start_pts = x[sampled_indices_point1]
        end_pts = x[sampled_indices_point2]
        segments_pts.append((start_pts, end_pts))

        # Create dataloader.
        print('===========================================')
        print('start_indices:', sampled_indices_point1, '\n', 'end_indices:', sampled_indices_point2)
        print('===========================================')

        dataset = CustomDataset(x0=torch.tensor(start_pts, dtype=torch.float32), 
                                x1=torch.tensor(end_pts, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

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
        distance_func_in_latent = lambda z: torch.exp(alpha * (1 - wd_model.positive_prob(z)))

        ofm = offmanifolder_maker_new(enc_func, disc_func, 
                                    disc_factor=args.disc_factor, 
                                    data_encodings=torch.tensor(x_encodings, dtype=torch.float32).to(device))
        # Visualize offmanifolder.
        ofm_scores = distance_func_in_latent(torch.tensor(x_encodings, dtype=torch.float32).to(device)).cpu().detach().numpy()
        ofm_scores_noisy = distance_func_in_latent(torch.tensor(x_noisy, dtype=torch.float32).to(device)).cpu().detach().numpy()
        ofm_scores_uniform = distance_func_in_latent(torch.tensor(uniform_samples, dtype=torch.float32).to(device)).cpu().detach().numpy()
        # fig = go.Figure()
        # fig.add_trace(go.Scatter3d(x=x_encodings[:,0], y=x_encodings[:,1], z=x_encodings[:,2], 
        #                            mode='markers', marker=dict(size=2, color=ofm_scores, colorscale='Viridis', opacity=0.8, colorbar=dict(title='Distance')),
        #                            hoverinfo='text',
        #                            text=ofm_scores))
        # fig.add_trace(go.Scatter3d(x=x_noisy[:,0], y=x_noisy[:,1], z=x_noisy[:,2],
        #                            mode='markers', marker=dict(size=2, color=ofm_scores_noisy, colorscale='Viridis', opacity=0.8, colorbar=dict(title='Distance')),
        #                            hovertext=ofm_scores_noisy))
        # fig.add_trace(go.Scatter3d(x=uniform_samples[:,0], y=uniform_samples[:,1], z=uniform_samples[:,2],
        #                            mode='markers', marker=dict(size=2, color=ofm_scores_uniform, colorscale='Viridis', opacity=0.8, colorbar=dict(title='Distance')),
        #                            hovertext=ofm_scores_uniform))
        # fig.update_layout(title='Distance to the manifold (extended dim) with alpha = %f' % alpha)
        # if args.show_plot:
        #     fig.show()
        # draw in plt
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        sc1 = ax.scatter(x_encodings[:,0], x_encodings[:,1], x_encodings[:,2], c=ofm_scores, cmap='viridis', alpha=0.8)
        sc2 = ax.scatter(x_noisy[:,0], x_noisy[:,1], x_noisy[:,2], c=ofm_scores_noisy, cmap='viridis', alpha=0.6)
        sc3 = ax.scatter(uniform_samples[:,0], uniform_samples[:,1], uniform_samples[:,2], c=ofm_scores_uniform, cmap='viridis', alpha=0.4)
        # add colorbar for all scatter plots
        cbar = fig.colorbar(sc3, ax=ax)
        cbar.set_label('Distance')

        ax.set_title('Distance to the manifold (extended dim) with alpha = %f' % alpha)
        plt.savefig(os.path.join(args.plots_save_dir, 'extended_dim_ofm.png'))
        
        print('Mean ext-dim true data: ', ofm_scores.mean(), 
            '\nMean ext-dim negative data: ', ofm_scores_noisy.mean(), 
            '\nMean ext-dim uniform data: ', ofm_scores_uniform.mean())

        # Visualize start/end points in latent space.
        # start_pts_encodings = enc_func(torch.tensor(start_pts, dtype=torch.float32).to(device)).cpu().detach().numpy()
        # end_pts_encodings = enc_func(torch.tensor(end_pts, dtype=torch.float32).to(device)).cpu().detach().numpy()
        # start_pts_encodings = encode_data(start_pts, ae_model.encoder, device)
        # end_pts_encodings = encode_data(end_pts, ae_model.encoder, device)
        # fig = go.Figure()
        # fig.add_trace(go.Scatter3d(x=x_encodings[:,0], y=x_encodings[:,1], z=x_encodings[:,2], 
        #                            mode='markers', marker=dict(size=2, color='gray', colorscale='Viridis', opacity=0.8)))
        # fig.add_trace(go.Scatter3d(x=start_pts_encodings[:,0], y=start_pts_encodings[:,1], z=start_pts_encodings[:,2], 
        #                            mode='markers', marker=dict(size=5, color='blue', colorscale='Viridis', opacity=0.8)))
        # fig.add_trace(go.Scatter3d(x=end_pts_encodings[:,0], y=end_pts_encodings[:,1], z=end_pts_encodings[:,2], 
        #                            mode='markers', marker=dict(size=5, color='red', colorscale='Viridis', opacity=0.8)))
        # if args.show_plot:
        #     fig.show()

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
            data_pts_encodings=torch.tensor(x_encodings, dtype=torch.float32).to(device),
            use_density=args.use_density,
            data_pts=torch.tensor(x, dtype=torch.float32).to(device),
            density_weight=args.density_weight,
            fixed_pot=args.fixed_pot,
            visualize_training=args.visualize_training,
            dataloader=dataloader,
            device=device,
            training_save_dir=args.training_save_dir,
        )

        # Train the model
        early_stopping = pl.callbacks.EarlyStopping(monitor='train_loss_epoch', patience=args.patience, mode='min')
        model_checkpoint = pl.callbacks.ModelCheckpoint(monitor='train_loss_epoch', save_top_k=1, mode='min', 
                                                        dirpath=args.checkpoint_dir, filename='gbmodel')
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            log_every_n_steps=args.log_every_n_steps,
            accelerator=device,
            callbacks=[early_stopping, model_checkpoint]
        )

        trainer.fit(gbmodel, train_dataloaders=dataloader)

        local_gbmodels.append(gbmodel)

    # Use Neural ODE to integrate the learned flow/vector field for each local vector field.
    print('=========== Running ODE on learned vector field ==========')
    adjoint = False
    if adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    local_trajs = []
    local_trajs_enc = []
    local_trajs_enc_unflat = []
    n_local_flows = len(local_gbmodels)

    for i, segment in enumerate(segments_pts):
        start_pts, end_pts = segment
        print('start_pts: ', start_pts.shape, '\n', 'end_pts: ', end_pts.shape)

        gbmodel = local_gbmodels[i]
        flow_ode = ODEFuncWrapper(gbmodel.flow_model).to('cpu') # (n_samples, n_features) -> (n_samples, n_features)

        n_samples = min(100, start_pts.shape[0])
        sampled_starts = torch.tensor(start_pts[:n_samples], dtype=torch.float32).to('cpu')
        with torch.no_grad():
            traj = odeint(flow_ode, sampled_starts, gbmodel.ts.to('cpu')) # [n_tsteps, n_samples, ambient_dim]
            print('inside traj: ', traj.shape)
        print('Flow Matching ODE Trajectory shape: ', traj.shape)
        encoded_traj = encode_data(traj.flatten(0,1), ae_model.encoder, 'cpu') # [n_tsteps*n_samples, latent_dim]
        print('Encoded Trajectory shape: ', encoded_traj.shape)
        local_trajs.append(traj)
        local_trajs_enc.append(encoded_traj)
        local_trajs_enc_unflat.append(encoded_traj.reshape(-1, n_samples, x_encodings.shape[1])) # [n_tsteps, n_samples, latent_dim]
    global_traj = np.concatenate(local_trajs, axis=0) # [n_tsteps * n_local_flows, n_samples, ambient_dim]
    global_traj_enc = np.concatenate(local_trajs_enc, axis=0) # [n_tsteps * n_samples * n_local_flows, latent_dim]
    global_traj_enc_unflat = np.concatenate(local_trajs_enc_unflat, axis=0) # [n_tsteps * n_local_flows, n_samples, latent_dim]
    print('Global Trajectory shape: ', global_traj.shape)
    print('Global Trajectory encoded shape: ', global_traj_enc.shape)
    print('Global Trajectory encoded unflat shape: ', global_traj_enc_unflat.shape)

    # Visualize the trajectory in latent space.
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    start_pts_enc = encode_data(start_pts, ae_model.encoder, 'cpu')
    end_pts_enc = encode_data(end_pts, ae_model.encoder, 'cpu') 
    ax.scatter(x_encodings[:,0], x_encodings[:,1], x_encodings[:,2], c='gray', alpha=0.8)
    ax.scatter(start_pts_enc[:,0], start_pts_enc[:,1], start_pts_enc[:,2], c='blue', alpha=0.8)
    ax.scatter(end_pts_enc[:,0], end_pts_enc[:,1], end_pts_enc[:,2], c='red', alpha=0.8)
    ax.scatter(global_traj_enc[:,0], global_traj_enc[:,1], global_traj_enc[:,2], c='blue', alpha=0.8)
    ax.set_title('ODE Trajectory in Latent Space')
    plt.savefig(os.path.join(args.plots_save_dir, 'ODE_latent_traj.png'))

    # plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x_encodings[:,0], y=x_encodings[:,1], z=x_encodings[:,2], 
                               mode='markers', marker=dict(size=2, color='gray', colorscale='Viridis', opacity=0.8)))
    fig.add_trace(go.Scatter3d(x=start_pts_enc[:,0], y=start_pts_enc[:,1], z=start_pts_enc[:,2], 
                               mode='markers', marker=dict(size=5, color='blue', colorscale='Viridis', opacity=0.8)))
    fig.add_trace(go.Scatter3d(x=end_pts_enc[:,0], y=end_pts_enc[:,1], z=end_pts_enc[:,2], 
                               mode='markers', marker=dict(size=5, color='red', colorscale='Viridis', opacity=0.8)))
    for i in range(n_samples):
        fig.add_trace(go.Scatter3d(x=global_traj_enc_unflat[:,i,0], y=global_traj_enc_unflat[:,i,1], z=global_traj_enc_unflat[:,i,2], 
                                   mode='lines', line=dict(width=2, color='blue')))
    fig.update_layout(title='ODE Trajectory in Latent Space')
    fig.write_html(os.path.join(args.plots_save_dir, 'ODE_latent_traj.html'))

        
    # Train a global Flow Matching model
    print('=========== Training Global Flow Matching Model ==========')
    n_local_flows = len(local_gbmodels)
    dataset = FlowMatchingDataset(torch.tensor(global_traj, dtype=torch.float32).to(device),
                                  args.n_tsteps, 
                                  n_local_flows)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    global_fm = FlowMatching(
        input_dim=x.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        local_flow_models=local_gbmodels,
        local_n_tsteps=args.n_tsteps,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # global_fm.astype(torch.float32)
    global_fm.to(device)
    global_fm.train()
    # freeze the parameters of the local gbmodels
    for gbmodel in local_gbmodels:
        for param in gbmodel.flow_model.parameters():
            param.requires_grad = False
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        accelerator=device,
    )
    trainer.fit(global_fm, train_dataloaders=dataloader)

    # Use Neural ODE to integrate the learned flow/vector field for the global flow matching model.
    print('=========== Running ODE on global flow vector field ==========')
    global_fm_ode = ODEFuncWrapper(global_fm).to('cpu') # (n_samples, n_features) -> (n_samples, n_features)
    n_samples = min(100, start_pts.shape[0])
    global_starts = segments_pts[0][0]
    global_mid = segments_pts[1][0]
    global_ends = segments_pts[1][1]
    print('sampled globalstarts: ', sampled_starts.shape)
    global_fm_trajs = []
    for i in range(n_local_flows):
        if i == 0:
            sampled_starts = torch.tensor(global_starts[:n_samples], dtype=torch.float32).to('cpu')
            ts = torch.linspace(0, 1, args.n_tsteps)
        elif i == n_local_flows - 1:
            sampled_starts = torch.tensor(global_mid[:n_samples], dtype=torch.float32).to('cpu')
            ts = torch.linspace(1, 2, args.n_tsteps)
        with torch.no_grad():
            cur_traj = odeint(global_fm_ode, sampled_starts, ts) # [n_tsteps, n_samples, ambient_dim]
        global_fm_trajs.append(cur_traj) 
    global_fm_traj = np.concatenate(global_fm_trajs, axis=0) # [n_tsteps * n_local_flows, n_samples, ambient_dim]
    print('Global FM ODE Trajectory shape: ', global_fm_traj.shape)
    global_t_steps = global_fm_traj.shape[0]

    # Visualize the trajectory in latent space.
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    global_starts_enc = encode_data(global_starts, ae_model.encoder, 'cpu')
    global_ends_enc = encode_data(global_ends, ae_model.encoder, 'cpu') 
    global_fm_traj_flattened = global_fm_traj.reshape(-1, x.shape[1]) # [n_tsteps * n_local_flows * n_samples, ambient_dim]
    global_fm_traj_enc_flattened = encode_data(global_fm_traj_flattened, ae_model.encoder, 'cpu')
    ax.scatter(x_encodings[:,0], x_encodings[:,1], x_encodings[:,2], c='gray', alpha=0.8)
    ax.scatter(global_starts_enc[:,0], global_starts_enc[:,1], global_starts_enc[:,2], c='blue', alpha=0.8)
    ax.scatter(global_ends_enc[:,0], global_ends_enc[:,1], global_ends_enc[:,2], c='red', alpha=0.8)
    ax.scatter(global_fm_traj_enc_flattened[:,0], global_fm_traj_enc_flattened[:,1], global_fm_traj_enc_flattened[:,2], c='blue', alpha=0.8)
    ax.set_title('Global FM ODE Trajectory in Latent Space')
    plt.savefig(os.path.join(args.plots_save_dir, 'global_fm_ODE_latent_traj.png'))

    # plotly
    global_fm_traj_enc = global_fm_traj_enc_flattened.reshape(global_t_steps, -1, x_encodings.shape[1]) # [n_tsteps * n_local_flows, n_samples, latent_dim]
    print('global_fm_traj_enc: ', global_fm_traj_enc.shape)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x_encodings[:,0], y=x_encodings[:,1], z=x_encodings[:,2], 
                               mode='markers', marker=dict(size=2, color='gray', colorscale='Viridis', opacity=0.8)))
    fig.add_trace(go.Scatter3d(x=global_starts_enc[:,0], y=global_starts_enc[:,1], z=global_starts_enc[:,2], 
                               mode='markers', marker=dict(size=5, color='blue', colorscale='Viridis', opacity=0.8)))
    fig.add_trace(go.Scatter3d(x=global_ends_enc[:,0], y=global_ends_enc[:,1], z=global_ends_enc[:,2], 
                               mode='markers', marker=dict(size=5, color='red', colorscale='Viridis', opacity=0.8)))
    for i in range(global_fm_traj_enc.shape[1]):
        fig.add_trace(go.Scatter3d(x=global_fm_traj_enc[:,i, 0], y=global_fm_traj_enc[:,i, 1], z=global_fm_traj_enc[:,i,2], 
                                   mode='lines', line=dict(width=2, color='blue')))
    fig.update_layout(title='Global FM ODE Trajectory in Latent Space')
    fig.write_html(os.path.join(args.plots_save_dir, 'global_fm_ODE_latent_traj.html'))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Discriminator Script")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--root_dir", type=str, default="../../", help="Root directory")
    parser.add_argument("--ae_run_id", type=str, default='pzlwi6t6', help="Autoencoder run ID")
    parser.add_argument("--data_path", type=str, default='../../data/eb_subset_all.npz')
    # Start/End points arguments
    parser.add_argument("--start_group", type=int, default=None, help="Start group for trajectory")
    parser.add_argument("--end_group", type=int, default=None, help="End group for trajectory")
    parser.add_argument("--start_idx", type=int, default=736, help="Start index for trajectory")
    parser.add_argument("--mid_idx", type=tuple, default=1553, help="Mid point index for trajectory")
    parser.add_argument("--end_idx", type=int, default=2543, help="End index for trajectory")
    parser.add_argument("--range_size", type=float, default=0.3, help="Range size for sampling")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    # GeodesicFM arguments
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
    parser.add_argument("--training_save_dir", type=str, default="./eb_local_fm/training/", help="Save directory")
    parser.add_argument("--patience", type=int, default=150, help="Patience for early stopping")
    parser.add_argument("--max_epochs", type=int, default=300, help="Maximum number of epochs")
    parser.add_argument("--log_every_n_steps", type=int, default=20, help="Log every n steps")
    parser.add_argument("--checkpoint_dir", type=str, default="./eb_fm/checkpoints", help="Checkpoint directory")
    parser.add_argument("--plots_save_dir", type=str, default="./eb_local_fm/plots", help="Save directory")
    parser.add_argument("--show_plot", action='store_true', help="Show plot")

    # Add new arguments for discriminator training
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
    # Function space regularized discriminator arguments
    parser.add_argument("--disc_use_function_space", action='store_true', help="Use function space for discriminator")
    parser.add_argument("--disc_fs_uniform_range", type=float, default=5.0, help="Uniform range for function space")
    parser.add_argument("--disc_fs_ce_loss_weight", type=float, default=1.0, help="CE loss weight for function space")
    parser.add_argument("--disc_fs_l2_loss_weight", type=float, default=1.0, help="L2 loss weight for function space")
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

    print('args: ', args)
    main(args)

