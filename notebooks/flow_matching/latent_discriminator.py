import argparse
import os
import sys
import numpy as np
import torch
import pytorch_lightning as pl
from glob import glob
import plotly.graph_objs as go
from torch.utils.data import Dataset, DataLoader
import torchdiffeq
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

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
    def __init__(self, in_dim, layer_widths=[64, 64, 64], activation='relu', loss_type='bce', normalize=False,
                 data_pts=None, k=5,
                 batch_norm=False, dropout=0.0, use_spectral_norm=False):
        super().__init__()
        self.in_dim = in_dim if data_pts is None else in_dim + k
        self.mlp = MLP(self.in_dim, 2, layer_widths, activation, batch_norm, dropout, use_spectral_norm)
        self.loss_type = loss_type
        self.normalize = normalize
        
        # hyperparameters for augmenting data with density feautres
        self.data_pts = data_pts
        self.k = int(k)
        
        self.lr = kwargs.get('lr', 1e-3)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)

        self.mean = torch.tensor(kwargs.get('mean', np.zeros(in_dim)), dtype=torch.float32)
        self.std = torch.tensor(kwargs.get('std', np.ones(in_dim)), dtype=torch.float32)
        
        self.train_step_outs = []
        self.val_step_outs = []
        self.test_step_outs = []
        self.train_ys = []
        self.val_ys = []
        self.test_ys = []
        
        if self.data_pts is not None:
            assert self.data_pts.shape[1] == self.in_dim
    
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
    model = Discriminator(in_dim=x.shape[1], layer_widths=args.disc_hidden_dim, activation='relu', 
                            loss_type='bce', normalize=False,
                            data_pts=data_pts, k=args.disc_density_k,
                            batch_norm=True, dropout=0.5, use_spectral_norm=True)
    
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
    def _forward_diffusion(x0, t, num_steps, beta_start, beta_end):
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

def sample_indices_within_range(points, selected_idx=None, range_size=0.1, num_samples=20, seed=23):
    # Implementation of sample_indices_within_range function
    # ...

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

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'

    # Load models
    ae_model = load_autoencoder(args.ae_run_id, args.root_dir)
    old_wd_model = load_discriminator(args.wd_run_id, args.root_dir)

    # Load data
    x, labels = load_data(args.data_path)
    x_encodings = encode_data(x, ae_model.encoder, device)

    # Negative sampling    
    if args.neg_method == 'add':
        x_noisy = neg_sample_using_additive(x, args.noise_levels, noise_rate=args.noise_rate, seed=args.seed, noise=args.noise_type)
    elif args.neg_method == 'diffusion':
        x_noisy = neg_sample_using_diffusion(x, args.t, args.num_steps, args.beta_start, args.beta_end, seed=args.seed)
    else:
        raise ValueError(f"Invalid negative sampling method: {args.neg_method}")
    assert x_noisy.shape[-1] == x_encodings.shape[-1]

    # Train new discriminator if args.train_discriminator is True
    if args.train_discriminator:
        print("Training new discriminator...")
        wd_model = train_discriminator(torch.tensor(x_encodings, dtype=torch.float32),
                                       torch.tensor(x_noisy, dtype=torch.float32),
                                       args)
    else:
        print("Using pre-trained discriminator...")

    # Select start/end points
    start_idx, sampled_indices_point1, end_idx, sampled_indices_point2 = sample_indices_within_range(
        encodings, selected_idx=(args.start_idx, args.end_idx), range_size=args.range_size, 
        seed=args.seed, num_samples=args.num_samples
    )

    # Create dataloader
    dataset = CustomDataset(x0=torch.tensor(x[sampled_indices_point1], dtype=torch.float32), 
                            x1=torch.tensor(x[sampled_indices_point2], dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Setup models for training
    ae_model = ae_model.to(device)
    old_wd_model = old_wd_model.to(device)

    enc_func = lambda x: ae_model.encoder(x)
    disc_func = lambda x: 1 - old_wd_model.positive_proba(enc_func(x))

    # Load or train discriminator
    if args.train_discriminator:
        print("Training new discriminator...")
        wd_model = train_discriminator(torch.tensor(encodings, dtype=torch.float32),
                                       torch.tensor(x_noisy, dtype=torch.float32),
                                       args)
    else:
        print("Loading pre-trained discriminator...")
        wd_model = load_discriminator(args.wd_run_id, args.root_dir)

    if args.use_new_discriminator:
        disc_func = lambda x: 1 - torch.sigmoid(wd_model(enc_func(x)))
    else:
        disc_func = lambda x: 1 - old_wd_model.positive_proba(enc_func(x))

    ofm, extended_dim_func = offmanifolder_maker_new(enc_func, disc_func, disc_factor=args.disc_factor, 
                                                     data_encodings=torch.tensor(encodings, dtype=torch.float32).to(device))

    gbmodel = GeodesicFM(
        func=ofm,
        encoder=enc_func,
        input_dim=x.shape[1],
        hidden_dim=args.hidden_dim, 
        scale_factor=args.scale_factor, 
        symmetric=args.symmetric, 
        num_layers=args.num_layers, 
        n_tsteps=args.n_tsteps, 
        lr=args.lr,
        weight_decay=args.weight_decay,
        flow_weight=args.flow_weight,
        length_weight=args.length_weight,
        cc_k=args.cc_k,
        density_weight=args.density_weight,
        data_pts=torch.tensor(x, dtype=torch.float32).to(device),
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

    # Generate and visualize trajectories
    # ... (implement trajectory generation and visualization)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Discriminator Script")
    parser.add_argument("--root_dir", type=str, default="../../", help="Root directory")
    parser.add_argument("--ae_run_id", type=str, default='pzlwi6t6', help="Autoencoder run ID")
    parser.add_argument("--wd_run_id", type=str, default='kafcutw4', help="Discriminator run ID")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data file")
    parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.2], help="Noise levels for negative sampling")
    parser.add_argument("--noise_type", type=str, default="gaussian", help="Type of noise for negative sampling")
    parser.add_argument("--start_idx", type=int, default=736, help="Start index for trajectory")
    parser.add_argument("--end_idx", type=int, default=2543, help="End index for trajectory")
    parser.add_argument("--range_size", type=float, default=0.3, help="Range size for sampling")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--disc_factor", type=float, default=5, help="Discriminator factor")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for GeodesicFM")
    parser.add_argument("--scale_factor", type=float, default=1, help="Scale factor for GeodesicFM")
    parser.add_argument("--symmetric", type=bool, default=True, help="Symmetric flag for GeodesicFM")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers for GeodesicFM")
    parser.add_argument("--n_tsteps", type=int, default=100, help="Number of time steps for GeodesicFM")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--flow_weight", type=float, default=0, help="Flow weight for GeodesicFM")
    parser.add_argument("--length_weight", type=float, default=1, help="Length weight for GeodesicFM")
    parser.add_argument("--cc_k", type=int, default=5, help="cc_k parameter for GeodesicFM")
    parser.add_argument("--density_weight", type=float, default=1., help="Density weight for GeodesicFM")
    parser.add_argument("--patience", type=int, default=150, help="Patience for early stopping")
    parser.add_argument("--max_epochs", type=int, default=300, help="Maximum number of epochs")
    parser.add_argument("--log_every_n_steps", type=int, default=20, help="Log every n steps")
    parser.add_argument("--checkpoint_dir", type=str, default="./eb_fm/checkpoints", help="Checkpoint directory")

    # Add new arguments for discriminator training
    parser.add_argument("--train_discriminator", action="store_true", help="Train a new discriminator")
    parser.add_argument("--use_new_discriminator", action="store_true", help="Use the newly trained discriminator")
    parser.add_argument("--disc_hidden_dim", type=int, default=64, help="Hidden dimension for discriminator")
    parser.add_argument("--disc_num_layers", type=int, default=3, help="Number of layers for discriminator")
    parser.add_argument("--disc_lr", type=float, default=1e-4, help="Learning rate for discriminator")
    parser.add_argument("--disc_epochs", type=int, default=100, help="Number of epochs for discriminator training")
    parser.add_argument("--disc_critic_iters", type=int, default=5, help="Number of critic iterations per epoch")
    parser.add_argument("--disc_batch_size", type=int, default=64, help="Batch size for discriminator training")
    parser.add_argument("--disc_weight_clip", type=float, default=0.01, help="Weight clipping value for discriminator")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")

    # Negative sampling arguments
    parser.add_argument("--neg_method", type=str, default="additive", help="additive|diffusion")
    parser.add_argument("--noise_rate", type=float, default=1.0, help="std of the additive noise")
    parser.add_argument("--t", type=int, default=100, help="Number of time steps for forward diffusion")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps for forward diffusion")
    parser.add_argument("--beta_start", type=float, default=0.0001, help="Start value for beta in forward diffusion")
    parser.add_argument("--beta_end", type=float, default=0.01, help="End value for beta in forward diffusion")


    args = parser.parse_args()
    main(args)