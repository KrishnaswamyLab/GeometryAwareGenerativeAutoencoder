import argparse
import os
import sys
from glob import glob
import plotly
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

def normalize(values):
    from sklearn.preprocessing import MinMaxScaler
    """
    Normalize a list or NumPy array of values (including negative) to the range [0, 1] using MinMaxScaler.

    Args:
    values (list or np.ndarray): A list or NumPy array of values.

    Returns:
    np.ndarray: A NumPy array of normalized values in the range [0, 1].
    """
    values = np.array(values).reshape(-1, 1)  # Reshape for the scaler
    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(values)
    return normalized_values.flatten()  # Flatten back to 1D array

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
            print('Initializing new GP discriminator... (no training needed)')
            gp_model = init_gp(torch.tensor(x_encodings, dtype=torch.float32), torch.ones(len(x_encodings), dtype=torch.float32).to(device))
        else:
            raise ValueError("Function space regularized discriminator not implemented.")

    # Visualize delta_y slice of the latent space, colored by the discriminator's positive probability.
    latent_dim = x_encodings.shape[1]
    r = args.disc_fs_uniform_range
    uniform_samples = np.random.uniform(-r, r, size=(300000, latent_dim))
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
    # Predict uncertainty from GP model.
    gp_model.eval()
    gp_model.to(device)
    pos_variances = gp_model.predict_uncertainty(torch.tensor(x_encodings, dtype=torch.float32).to(device)).cpu().detach().numpy()
    neg_variances = gp_model.predict_uncertainty(torch.tensor(x_noisy, dtype=torch.float32).to(device)).cpu().detach().numpy()
    uniform_variances = gp_model.predict_uncertainty(torch.tensor(uniform_samples, dtype=torch.float32).to(device)).cpu().detach().numpy()
    print('pos_variances: ', pos_variances.mean())
    print('neg_variances: ', neg_variances.mean())
    print('uniform_variances: ', uniform_variances.mean())
    normalized_pos_variances = normalize(np.log(pos_variances))
    normalized_neg_variances = normalize(np.log(neg_variances))
    normalized_uniform_variances = normalize(np.log(uniform_variances))
    print('normalized_(log)pos_variances: ', normalized_pos_variances.mean())
    print('normalized_(log)neg_variances: ', normalized_neg_variances.mean())
    print('normalized_(log)uniform_variances: ', normalized_uniform_variances.mean())    


    # Compute extended dimension for each sets.
    alpha = args.alpha
    distance_func_in_latent = lambda z: torch.exp(alpha * (1 - wd_model.positive_prob(z)))
    pos_extended_dim = distance_func_in_latent(torch.tensor(x_encodings, dtype=torch.float32).to(device)).cpu().detach().numpy()
    neg_extended_dim = distance_func_in_latent(torch.tensor(x_noisy, dtype=torch.float32).to(device)).cpu().detach().numpy()
    uniform_extended_dim = distance_func_in_latent(torch.tensor(uniform_samples, dtype=torch.float32).to(device)).cpu().detach().numpy()
    
    gp_distance_func = lambda v: torch.exp(alpha * (1 - (1 - v)))
    gp_pos_extended_dim = gp_distance_func(torch.tensor(normalized_pos_variances, dtype=torch.float32).to(device)).cpu().detach().numpy()
    gp_neg_extended_dim = gp_distance_func(torch.tensor(normalized_neg_variances, dtype=torch.float32).to(device)).cpu().detach().numpy()
    gp_uniform_extended_dim = gp_distance_func(torch.tensor(normalized_uniform_variances, dtype=torch.float32).to(device)).cpu().detach().numpy()
    print('gp_pos_extended_dim: ', gp_pos_extended_dim.mean())
    print('gp_neg_extended_dim: ', gp_neg_extended_dim.mean())
    print('gp_uniform_extended_dim: ', gp_uniform_extended_dim.mean())

    delta_y = 0.1
    for y in np.arange(-1, 1, delta_y):
        c = delta_y / 2 # interval width
        pos_indices = np.where((x_encodings[:, 1] >= y-c) & (x_encodings[:, 1] <= y+c))[0]
        print('pos_indices: ', len(pos_indices))
        neg_indices = np.where((x_noisy[:, 1] >= y-c) & (x_noisy[:, 1] <= y+c))[0]
        print('neg_indices: ', len(neg_indices))
        uniform_indices = np.where((uniform_samples[:, 1] >= y-c) & (uniform_samples[:, 1] <= y+c))[0]
        print('uniform_indices: ', len(uniform_indices))

        # Discriminator prediction vs. GP prediction side by side.
        fig = plt.figure(figsize=(8 * 2, 6))
        ax = fig.add_subplot(121)
        # pc = ax.scatter(x_encodings[pos_indices,0], x_encodings[pos_indices,2], c=pos_probs[pos_indices], cmap='viridis', alpha=1, marker='o')
        # ax.scatter(x_noisy[neg_indices,0], x_noisy[neg_indices,2], c=neg_probs[neg_indices], cmap='viridis', alpha=0.6, marker='o')
        pc = ax.scatter(x_encodings[pos_indices,0], x_encodings[pos_indices,2], c='red', cmap='viridis', alpha=1, marker='x', label=f'y in [{(y-c):.2f}, {(y+c):.2f}]')
        pc = ax.scatter(uniform_samples[uniform_indices,0], uniform_samples[uniform_indices,2], c=uniform_probs[uniform_indices], 
                        cmap='viridis', alpha=1, marker='o', edgecolors='black')
        plt.colorbar(pc, label='Pred Probability')
        # ax.set_xlim(-r, r)
        # ax.set_ylim(-r, r)
        ax.grid(True)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title(f'y in [{(y-c):.2f}, {(y+c):.2f}]')

        ax = fig.add_subplot(122)
        pc = ax.scatter(x_encodings[pos_indices,0], x_encodings[pos_indices,2], c='red', cmap='viridis', alpha=1, marker='x')
        variance_power = 2
        pc = ax.scatter(uniform_samples[uniform_indices,0], uniform_samples[uniform_indices,2], c=(1-normalized_uniform_variances[uniform_indices])**variance_power, 
                   cmap='viridis', alpha=1.0, marker='o', edgecolors='black')
        plt.colorbar(pc, label='(1-normalized(log(variance)))**power')
        ax.grid(True)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title(f'y in [{(y-c):.2f}, {(y+c):.2f}]')
        plt.savefig(os.path.join(args.plots_save_dir, f'latent_space_y_centered_{y:.2f}.png'))


        # ============================================================
        # Visualize extended dimension.
        fig = plt.figure(figsize=(8*2, 6))
        ax = fig.add_subplot(121)
        pc = ax.scatter(x_encodings[pos_indices,0], x_encodings[pos_indices,2], c='red', cmap='viridis', alpha=1, marker='x')
        pc = ax.scatter(uniform_samples[uniform_indices,0], uniform_samples[uniform_indices,2], c=uniform_extended_dim[uniform_indices], 
                   cmap='viridis', alpha=1.0, marker='o', edgecolors='black')
        #ax.scatter(x_noisy[neg_indices,0], x_noisy[neg_indices,2], c='blue', cmap=neg_extended_dim[neg_indices], alpha=1, marker='o', edgecolors='black')
        plt.colorbar(pc, label='exp(alpha * (1 - positive_prob))')
        ax.grid(True)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title(f'Extended dimension in [{(y-c):.2f}, {(y+c):.2f}]')

        ax = fig.add_subplot(122)
        pc = ax.scatter(x_encodings[pos_indices,0], x_encodings[pos_indices,2], c='red', cmap='viridis', alpha=1, marker='x')
        pc = ax.scatter(uniform_samples[uniform_indices,0], uniform_samples[uniform_indices,2], c=gp_uniform_extended_dim[uniform_indices], 
                   cmap='viridis', alpha=1.0, marker='o', edgecolors='black')
        plt.colorbar(pc, label=f'exp(alpha * (normalized(log(variance))))')
        ax.grid(True)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title(f'Extended dimension in [{(y-c):.2f}, {(y+c):.2f}]')
        plt.savefig(os.path.join(args.plots_save_dir, f'extended_dim_y_centered_{y:.2f}.png'))


    # Visualize discriminator positive probs prediction on positive and negative samples.
    # Uniform sample N points from {-r, r} ^ latent_dim.
    # latent_dim = x_encodings.shape[1]
    # r = args.disc_fs_uniform_range
    # uniform_samples = np.random.uniform(-r, r, size=(1000, latent_dim))
    # wd_model.eval()
    # wd_model.to(device)
    # with torch.no_grad():
    #     pos_probs = wd_model.positive_prob(torch.tensor(x_encodings, dtype=torch.float32).to(device)).cpu().detach().numpy()
    #     neg_probs = wd_model.positive_prob(torch.tensor(x_noisy, dtype=torch.float32).to(device)).cpu().detach().numpy()
    #     uniform_probs = wd_model.positive_prob(torch.tensor(uniform_samples, dtype=torch.float32).to(device)).cpu().detach().numpy()
    #     if hasattr(wd_model, 'classify'):
    #         labels_pred = wd_model.classify(torch.tensor(x, dtype=torch.float32).to(device)).cpu().detach().numpy()
    # print('pos_probs: ', pos_probs.mean())
    # print('neg_probs: ', neg_probs.mean())
    # print('uniform_probs: ', uniform_probs.mean())
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Latent Discriminator Script")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--root_dir", type=str, default="../../", help="Root directory")
    parser.add_argument("--ae_run_id", type=str, default='pzlwi6t6', help="Autoencoder run ID")
    parser.add_argument("--data_path", type=str, default='../../data/eb_subset_all.npz')
    # Start/End points arguments
    parser.add_argument("--start_group", type=int, default=None, help="Start group for trajectory")
    parser.add_argument("--end_group", type=int, default=None, help="End group for trajectory")
    parser.add_argument("--start_idx", type=int, default=736, help="Start index for trajectory")
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
    parser.add_argument("--training_save_dir", type=str, default="./eb_fm/training/", help="Save directory")
    parser.add_argument("--patience", type=int, default=150, help="Patience for early stopping")
    parser.add_argument("--max_epochs", type=int, default=300, help="Maximum number of epochs")
    parser.add_argument("--log_every_n_steps", type=int, default=20, help="Log every n steps")
    parser.add_argument("--checkpoint_dir", type=str, default="./eb_fm/checkpoints", help="Checkpoint directory")
    parser.add_argument("--plots_save_dir", type=str, default="./eb_fm/demon/", help="Save directory")
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
    parser.add_argument("--disc_fs_uniform_range", type=float, default=2.0, help="Uniform range for function space")
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
    args.plots_save_dir = f"Demo_GP_{args.disc_use_gp}-alpha_{args.alpha}-discfactor_{args.disc_factor}-fixedpot_{args.fixed_pot}-embedt_{args.embed_t}-{os.path.basename(args.plots_save_dir)}"

    print('args: ', args)
    main(args)
