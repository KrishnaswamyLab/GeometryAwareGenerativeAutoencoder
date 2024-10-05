import os
import numpy as np
import torch
import wandb
import anndata as ad
import scanpy as sc
import phate
import argparse
from scipy import sparse as sp
from scipy.spatial.distance import pdist, squareform
import pytorch_lightning as pl
from torch.nn.utils import spectral_norm

import matplotlib.pyplot as plt
import plotly.graph_objs as go

def convert_data(X, colors, seed=42, test_size=0.1, knn=5, t='auto', n_components=3):
    # if X is sparse, convert to dense
    if sp.issparse(X):
        X = X.toarray()
        
    phate_op = phate.PHATE(random_state=seed, t=t, n_components=n_components, knn=knn)
    phate_data = phate_op.fit_transform(X)

    dists = squareform(pdist(phate_op.diff_potential))

    return dict(
        data=X,
        colors=colors,
        dist=dists,
        phate=phate_data
    )

def prepare_EB_data(preprocessed_eb_path, ambient_dim=50, ambient_source='pca', latent_dim=3, save_dir='../../data/'):
    '''
    Args:
        preprocessed_eb_path: path to the preprocessed EB dataset, in .h5ad format; has obs['sample_labels']
        ambient_dim: dimension of the ambient space.
        ambient_source: source of the ambient space, either 'pca' or 'hvg'.
        latent_dim: dimension of the latent space.
    Returns:
        data with keys: 'data', 'dist', 'colors', 'phate'.
    '''
    save_path = os.path.join(save_dir, f'eb_D-{ambient_dim}_d-{latent_dim}_{ambient_source}.npz')

    data = None
    if os.path.exists(save_path):
        data = np.load(save_path, allow_pickle=True)
    else:
        eb_data = ad.read_h5ad(preprocessed_eb_path) # library size normalized, sqrt transformed, unwanted genes/cells removed.
        if ambient_source == 'pca':
            # compute PCA
            sc.tl.pca(eb_data, svd_solver='arpack', n_comps=ambient_dim)
            X = eb_data.obsm['X_pca']
        elif ambient_source == 'hvg':
            sc.pp.highly_variable_genes(eb_data, n_top_genes=ambient_dim)
            X = eb_data.X[:, eb_data.var['highly_variable']]
        else:
            raise ValueError(f'Invalid ambient source: {ambient_source}')
    
        # # Phate
        # phate_op = phate.PHATE(n_components=latent_dim, n_jobs=-2, random_state=42)
        # phate_coords = phate_op.fit_transform(data['data']) # [N, latent_dim]
        # data['phate'] = phate_coords

        # # Compute distance matrix
        # dist_mat = torch.cdist(torch.tensor(phate_coords), torch.tensor(phate_coords)).numpy()
        # data['dist'] = dist_mat

        # Colors
        sample_labels = eb_data.obs['sample_labels']
        unique_labels = np.unique(sample_labels)
        str2class = {label: i for i, label in enumerate(unique_labels)}
        colors =  np.array([str2class[label] for label in sample_labels])

        data = convert_data(X, colors)

        print('data', data['data'].shape, 'dist', data['dist'].shape, 'colors', data['colors'].shape, 'phate', data['phate'].shape)

        # Save
        np.savez(save_path, **data)

    # Visualize PHATE coords colored by labels.
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['phate'][:, 0], data['phate'][:, 1], data['phate'][:, 2], c=data['colors'], cmap='viridis')
    plt.savefig(os.path.join(save_dir, f'phate_coords_colored_by_labels.png'))

    # Plotly for PHATE coords colored by labels.
    fig = go.Figure()
    for label in np.unique(data['colors']):
        cur_phate = data['phate'][data['colors'] == label]
        fig.add_scatter3d(x=cur_phate[:, 0], y=cur_phate[:, 1], z=cur_phate[:, 2], mode='markers', name=str(label))
    fig.write_html(os.path.join(save_dir, f'phate_coords_colored_by_labels.html'))

    return data
    
    


class PointCloudDataset(torch.utils.data.Dataset):
    """
    Point Cloud Dataset
    """
    def __init__(self, pointcloud, distances):
        self.pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
        self.distances = torch.tensor(distances, dtype=torch.float32)

    def __len__(self):
        return len(self.pointcloud)
    
    def __getitem__(self, idx):
        return self.pointcloud[idx], idx  # Return point cloud and its index

def make_custom_collate_fn(dataset):
    def custom_collate_fn(batch):
        pointclouds, indices = zip(*batch)
        pointclouds = torch.stack(pointclouds)
        
        # Access `distances` from the dataset
        dist_mat = dataset.distances[torch.tensor(indices)][:, torch.tensor(indices)]
        dist_mat_upper_tri = dist_mat[np.triu_indices(dist_mat.size(0), k=1)]
        #print('custom_collate_fn: x', pointclouds.shape, 'd', dist_mat_upper_tri.shape, 'dist_mat.shape', dist_mat.shape)
        batch = {'x': pointclouds, 'd': dist_mat_upper_tri}
        return batch
    
    return custom_collate_fn

# class PointCloudDataset(torch.utils.data.Dataset):
#     """
#     Point Cloud Dataset
#     """
#     def __init__(self, pointcloud, distances, batch_size=64, shuffle=True):
#         self.pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
#         self.distances = torch.tensor(distances, dtype=torch.float32)
#         self.shuffle = shuffle
#         self.batch_size = batch_size

#         if shuffle:
#             self.idxs = torch.randperm(len(self.pointcloud))
#         else:
#             self.idxs = torch.arange(len(self.pointcloud))

#     def __len__(self):
#         return len(self.pointcloud)
    
#     def __getitem__(self, idx):
#         '''
#             Returns a batch of pointclouds and their distances
#             batch['x'] = [B, D]
#             batch['d'] = [B, B(B-1)/2] (upper triangular), assuming symmetric distance matrix
#         '''
#         if self.shuffle:
#             batch_idxs = self.idxs[:self.batch_size]
#         else:
#             batch_idxs = torch.arange(idx, idx+self.batch_size) % len(self.pointcloud)
#         batch = {}
#         batch['x'] = self.pointcloud[batch_idxs]
#         dist_mat = self.distances[batch_idxs][:,batch_idxs]
#         batch['d'] = dist_mat[np.triu_indices(dist_mat.size(0), k=1)]

#         return batch


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, layer_widths=[256, 128, 64], activation='relu', batch_norm=False, dropout=0.0, use_spectral_norm=False):
        super().__init__()


        # layer_widths = cfg.get("layer_widths", [64, 64, 64])
        # assert len(layer_widths) >= 2, "layer_widths list must contain at least 2 elements"
        # activation = cfg.get("activation", "relu")
        # assert activation in activation_dict.keys(), f"activation must be one of {list(activation_dict.keys())}"
        # batch_norm = cfg.get("batch_norm", False)
        # dropout = cfg.get("dropout", 0.0)
        # use_spectral_norm = cfg.get("spectral_norm", False)  # Configuration for using spectral normalization

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_widths = layer_widths
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.use_spectral_norm = use_spectral_norm

        layers = []
        for i, width in enumerate(layer_widths):
            if i == 0:  # First layer, input dimension to first layer width
                linear_layer = torch.nn.Linear(in_dim, width)
            else:  # Subsequent layers, previous layer width to current layer width
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

class Encoder(torch.nn.Module):
    def __init__(self, data_dim, latent_dim, layer_widths=[256, 128, 64], activation='relu', 
                 batch_norm=False, dropout=0.0, use_spectral_norm=False, 
                 mean=0, std=1, dist_std=1., dist_mse_decay=0.):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.mean = mean
        self.std = std
        self.dist_std = dist_std
        self.dist_mse_decay = dist_mse_decay
        self.mlp = MLP(data_dim, latent_dim, layer_widths=layer_widths, activation=activation, 
                       batch_norm=batch_norm, dropout=dropout, use_spectral_norm=use_spectral_norm)
    
    def _normalize(self, x):
        return (x - self.mean) / self.std
    
    def _normalize_dist(self, d):
        return d / self.dist_std

    def forward(self, x, normalize=True): # takes in unnormalized data.
        #print(x.shape)
        if normalize:
            x = self._normalize(x)
        return self.mlp(x)
    
    def loss_function(self, dist_gt_normalized, z): # assume normalized.
        '''
        Args:
            dist_gt_normalized: normalized ground truth distance matrix. [B * (B-1) / 2]
            z: latent embedding. [B, latent_dim]
        Returns:
            loss: scalar loss.
        '''
        if torch.backends.mps.is_available():
            dist_emb = torch.cdist(z, z, p=2)
            # take the upper triangular part of the distance matrix
            dist_emb = dist_emb[np.triu_indices(dist_emb.size(0), k=1)].flatten() # [B * (B-1) / 2]
        else:
            dist_emb = torch.nn.functional.pdist(z) # [B * (B-1) / 2]
        #print('dist_emb', dist_emb.shape, 'dist_gt_normalized', dist_gt_normalized.shape)
        if self.dist_mse_decay > 0.:
            return (torch.square(dist_emb - dist_gt_normalized) * torch.exp(-self.dist_mse_decay * dist_gt_normalized)).mean()
        else:
            return torch.nn.functional.mse_loss(dist_emb, dist_gt_normalized)

    def step(self, batch, batch_idx, stage):
        x = batch['x']
        d = batch['d']
        mask = batch.get('md', None)
        z = self.forward(x)
        d_norm = self.preprocessor.normalize_dist(d)
        loss = self.loss_function(d_norm, z, mask)
        self.log(f'{stage}/loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'validation')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'test')
        return loss
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #     return optimizer
    
class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, data_dim, layer_widths=[64, 128, 256], activation='relu', 
                 batch_norm=False, dropout=0.0, use_spectral_norm=False, 
                 mean=0, std=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.mean = mean
        self.std = std
        self.mlp = MLP(latent_dim, data_dim, layer_widths=layer_widths, activation=activation, 
                       batch_norm=batch_norm, dropout=dropout, use_spectral_norm=use_spectral_norm)
    
    def _normalize(self, x):
        # reverse normalization
        return (x * self.std) + self.mean

    def forward(self, z, unnormalize=False):
        x = self.mlp(z)
        if unnormalize: # outputs unnormalized data
            x = self._normalize(x)
        return x
    
    def loss_function(self, x_normalized, xhat_normalized):
        return torch.nn.functional.mse_loss(xhat_normalized, x_normalized)

class Autoencoder(pl.LightningModule):
    def __init__(self, data_dim, latent_dim, encoder_layer_width=[256, 128, 64], decoder_layer_width=[64, 128, 256], activation='relu', 
                 batch_norm=False, dropout=0.0, use_spectral_norm=False, 
                 mean=0, std=1, dist_std=1., dist_mse_decay=0.,
                 weights_dist=1, weights_reconstr=1, weights_cycle=0, weights_cycle_dist=0,
                 lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.encoder = Encoder(data_dim, latent_dim, layer_widths=encoder_layer_width, activation=activation, 
                 batch_norm=batch_norm, dropout=dropout, use_spectral_norm=use_spectral_norm, 
                 mean=mean, std=std, dist_std=dist_std, 
                 dist_mse_decay=dist_mse_decay)
        self.decoder = Decoder(latent_dim, data_dim, layer_widths=decoder_layer_width, activation=activation, 
                 batch_norm=batch_norm, dropout=dropout, use_spectral_norm=use_spectral_norm, 
                 mean=mean, std=std)
        self.weights_dist = weights_dist
        self.weights_reconstr = weights_reconstr
        self.weights_cycle = weights_cycle
        self.weights_cycle_dist = weights_cycle_dist

        self.mean = mean
        self.std = std
        self.dist_std = dist_std
        self.dist_mse_decay = dist_mse_decay

        self.lr = lr
        self.weight_decay = weight_decay

        self.save_hyperparameters() # save hyperparameters s.t. we can load them later using pl.Trainer.load_from_checkpoint.

    def forward(self, x):
        return self.decoder(self.encoder(x, normalize=True), unnormalize=False)
    
    def end2end_step(self, batch, batch_idx, stage):
        x = batch['x']
        d = batch['d']
        #print('2end2end_step: x', x.shape, 'd', d.shape)
        x_normalized = self.encoder._normalize(x)
        d_normalized = self.encoder._normalize_dist(d)
        z = self.encoder(x_normalized)
        xhat_normalized= self.decoder(z, unnormalize=False)
        loss = self.loss_function(xhat_normalized, x_normalized, z, d_normalized, stage)
        return loss

    def step(self, batch, batch_idx, stage):
        loss = self.end2end_step(batch, batch_idx, stage)
        self.log(f'{stage}/loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'test')
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


    def loss_function(self, xhat_norm, x_norm, z, d_norm, stage):
        '''
        Args:
            xhat_norm: normalized reconstructed data. [B, D]
            x_norm: normalized ground truth data. [B, D]
            z: latent embedding. [B, latent_dim]
            d_norm: normalized ground truth distance matrix. [B, (B-1)/2]
            stage: train/validation/test
        '''
        loss = 0.0
        assert self.weights_dist + self.weights_reconstr > 0.0, "At least one loss must be enabled"

        if self.weights_dist > 0.0:
            dl = self.encoder.loss_function(dist_gt_normalized=d_norm, z=z)
            self.log(f'{stage}/dist_loss', dl, prog_bar=True, on_epoch=True)
            loss += self.weights_dist * dl

        if self.weights_reconstr > 0.0:
            rl = self.decoder.loss_function(x_normalized=x_norm, xhat_normalized=xhat_norm)
            self.log(f'{stage}/reconstr_loss', rl, prog_bar=True, on_epoch=True)
            loss += self.weights_reconstr * rl

        if self.weights_cycle + self.weights_cycle_dist > 0.0:
            z2 = self.encoder(xhat_norm, normalize=False)
            if self.weights_cycle > 0.0:
                l_z_z2 = torch.nn.functional.mse_loss(z, z2)
                self.log(f'{stage}/cycle_loss', l_z_z2, prog_bar=True, on_epoch=True)
                loss += self.weights_cycle * l_z_z2
            if self.weights_cycle_dist > 0.0:
                l_d_z2 = self.encoder.loss_function(gt_dist_normalized=d_norm, z=z2)
                self.log(f'{stage}/cycle_dist_loss', l_d_z2, prog_bar=True, on_epoch=True)
                loss += self.weights_cycle_dist * l_d_z2
        return loss
    

def load_data(data_path):
    data = np.load(data_path)
    return data['data'], data['dist'], data['colors'], data['phate']

def split_pointcloud(pointcloud, distances, labels, train_ratio=0.9):
    # Split the data into training and test sets
    train_val_indices = np.random.choice(len(pointcloud), size=int(len(pointcloud) * train_ratio), replace=False)
    test_indices = np.setdiff1d(np.arange(len(pointcloud)), train_val_indices)

    # Split the data into training and validation sets
    train_indices = np.random.choice(train_val_indices, size=int(len(train_val_indices) * 0.85), replace=False)
    val_indices = np.setdiff1d(train_val_indices, train_indices)

    train_pointcloud = pointcloud[train_indices]
    val_pointcloud = pointcloud[val_indices]
    test_pointcloud = pointcloud[test_indices]
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    test_labels = labels[test_indices]

    # Split the distances matrix
    train_distances = distances[train_indices][:, train_indices]
    val_distances = distances[val_indices][:, val_indices]
    test_distances = distances[test_indices][:, test_indices]

    assert train_pointcloud.shape[0] == train_distances.shape[0] == len(train_indices)
    assert val_pointcloud.shape[0] == val_distances.shape[0] == len(val_indices)
    assert test_pointcloud.shape[0] == test_distances.shape[0] == len(test_indices)

    print(f"Train pointcloud: {train_pointcloud.shape}, validation pointcloud: {val_pointcloud.shape}, test pointcloud: {test_pointcloud.shape}")
    print(f"Train distances: {train_distances.shape}, validation distances: {val_distances.shape}, test distances: {test_distances.shape}")

    return train_pointcloud, train_distances, train_labels, val_pointcloud, val_distances, val_labels, test_pointcloud, test_distances, test_labels
    

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'

    # Load data and preprocess.
    if args.eb_h5ad_path:
        data_dict = prepare_EB_data(preprocessed_eb_path=args.eb_h5ad_path, 
                                    ambient_dim=args.ambient_dim, ambient_source=args.ambient_source, 
                                    latent_dim=args.latent_dim, save_dir=args.data_save_dir)
        pointcloud, distances, labels, phate_coords = data_dict['data'], data_dict['dist'], data_dict['colors'], data_dict['phate']
    else:
        data_dict = load_data(args.data_path)
        pointcloud, distances, labels, phate_coords = data_dict

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
        #axs[i].axis('off')
    plt.tight_layout()
    plt.savefig('./pointcloud_splits.png')
    #plt.show()

    # mean, std, dist_std
    component_wise_normalization = args.component_wise_normalization
    if component_wise_normalization:
        mean = pointcloud.mean(axis=0) # [D]
        std = pointcloud.std(axis=0) # [D]
    else:
        mean = pointcloud.mean() # scalar
        std = pointcloud.std() # scalar
    dist_std = np.std(distances.flatten()) # scalar
    # Init model.
    ae_model = Autoencoder(data_dim=pointcloud.shape[1], latent_dim=args.latent_dim, encoder_layer_width=args.encoder_layer_width, decoder_layer_width=args.decoder_layer_width, activation=args.activation, 
                           batch_norm=args.batch_norm, dropout=args.dropout, use_spectral_norm=args.use_spectral_norm, 
                           mean=mean, std=std, dist_std=dist_std,
                           dist_mse_decay=args.dist_mse_decay, lr=args.lr, weight_decay=args.weight_decay, 
                           weights_dist=args.weights_dist, weights_reconstr=args.weights_reconstr, 
                           weights_cycle=args.weights_cycle, weights_cycle_dist=args.weights_cycle_dist)
    
    # Train.
    early_stop = pl.callbacks.EarlyStopping(monitor='val/loss', patience=args.ae_early_stop_patience, mode='min')
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor='val/loss', save_top_k=1, mode='min', 
                                                    dirpath=args.checkpoint_dir, filename='autoencoder')
    
    trainer = pl.Trainer(max_epochs=args.ae_max_epochs, log_every_n_steps=args.ae_log_every_n_steps, 
                         callbacks=[early_stop, model_checkpoint])
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
    plt.savefig('./latent_space_reconstruction.png')

    # plotly visualization
    fig = go.Figure()
    for i, (data, labels) in enumerate([(train_pointcloud, train_labels), (val_pointcloud, val_labels), (test_pointcloud, test_labels)]):
        z = best_ae_model.encoder(torch.tensor(data, dtype=torch.float32, device=device), normalize=True)
        xhat_unnormalized = best_ae_model.decoder(z, unnormalize=True)
        z = z.detach().cpu().numpy()
        xhat_unnormalized = xhat_unnormalized.detach().cpu().numpy()
        fig.add_trace(go.Scatter3d(x=z[:, 0], y=z[:, 1], z=z[:, 2], mode='markers', marker=dict(size=2, color=labels, colorscale='viridis'), name=f'Latent Space of {i}'))
    fig.show()

def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'

    # Load data and preprocess.
    pointcloud, distances, labels, phate_coords = load_data(args.data_path)

    # Load pretrained autoencoder.
    best_ae_model = Autoencoder.load_from_checkpoint(args.checkpoint_dir + '/autoencoder.ckpt')
    best_ae_model.to(device)

    # Evaluate.
    zs = []
    bs = args.batch_size
    for i in range(0, pointcloud.shape[0], bs):
        x = torch.tensor(pointcloud[i:i+bs], dtype=torch.float32, device=device)
        z = best_ae_model.encoder(x, normalize=True)
        zs.append(z.detach().cpu().numpy())
    zs = np.concatenate(zs, axis=0)

    print('zs', zs.shape)

    return

if __name__ == "__main__":
    # Argparse 
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument("--mode", type=str, default='train', help="train|eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eb_h5ad_path", type=str, default=None, help="Path to the preprocessed EB data")
    parser.add_argument("--ambient_dim", type=int, default=50, help="Ambient dimension")
    parser.add_argument("--ambient_source", type=str, default='hvg', help="Ambient source")
    parser.add_argument("--data_save_dir", type=str, default='../../data/', help="Directory to save data")
    parser.add_argument("--data_path", type=str, default='../../data/eb_all.npz', help="Path to the data")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--ae_max_epochs", type=int, default=50, help="Maximum number of epochs for training")
    parser.add_argument("--ae_early_stop_patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--ae_log_every_n_steps", type=int, default=100, help="Log every n steps")
    parser.add_argument("--checkpoint_dir", type=str, default='./ae_checkpoints', help="Directory to save checkpoints")
    parser.add_argument("--latent_dim", type=int, default=3, help="Latent dimension")
    parser.add_argument("--encoder_layer_width", type=int, nargs='+', default=[256, 128, 64], help="Encoder layer widths")
    parser.add_argument("--decoder_layer_width", type=int, nargs='+', default=[64, 128, 256], help="Decoder layer widths")
    parser.add_argument("--activation", type=str, default='relu', help="Activation function")
    parser.add_argument("--batch_norm", action='store_true', help="Use batch normalization")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate") # TODO: might separate into encoder and decoder.
    parser.add_argument("--use_spectral_norm", action='store_true', help="Use spectral normalization")
    parser.add_argument("--dist_mse_decay", type=float, default=0.5, help="Decay rate for distance loss")
    parser.add_argument("--weights_dist", type=float, default=77.4, help="Weight for distance loss")
    parser.add_argument("--weights_reconstr", type=float, default=0.32, help="Weight for reconstruction loss")
    parser.add_argument("--weights_cycle", type=float, default=1, help="Weight for cycle loss")
    parser.add_argument("--weights_cycle_dist", type=float, default=0, help="Weight for cycle distance loss")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--component_wise_normalization", action='store_true', help="Use component-wise normalization")
    parser.add_argument("--wandb", action='store_true', help="Use wandb")
    parser.add_argument("--wandb_project", type=str, default='dmae', help="Wandb project")
    parser.add_argument("--wandb_entity", type=str, default='danqiliao', help="Wandb entity")

    args = parser.parse_args()

    if args.wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        wandb.config.update(args)

    if args.mode == 'train':
        main(args)
    elif args.mode == 'eval':
        evaluate(args)
    