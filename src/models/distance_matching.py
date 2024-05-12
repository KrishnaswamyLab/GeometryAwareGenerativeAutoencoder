import sys
import os
import numpy as np
import torch
import phate
import scipy
import matplotlib.pyplot as plt
import scprep

sys.path.append('../../src/')
from data import train_valid_loader_from_pc
from model2 import Encoder, Decoder, Preprocessor, Autoencoder
from models.unified_model import GeometricAE

from data_script import hemisphere_data, sklearn_swiss_roll
from utils.seed import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from omegaconf import OmegaConf

activation_dict = {
    'relu': torch.nn.ReLU(),
    'leaky_relu': torch.nn.LeakyReLU(),
    'sigmoid': torch.nn.Sigmoid()
}

def train_model(cfg, model, trainloader, valloader, logger, checkpoint_callback):
    early_stopping = EarlyStopping(cfg.training.monitor, patience=cfg.training.patience)
    trainer = Trainer(
        logger=logger,
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        callbacks=[early_stopping,checkpoint_callback],
        log_every_n_steps=cfg.training.log_every_n_steps,
    )

    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=valloader,
    )

class DistanceMatching(GeometricAE):
    ''' Geometric Autoencoder with distance matching encoding.'''
    def __init__(self, 
                 ambient_dimension, 
                 latent_dimension, 
                 model_type = 'distance', 
                 activation = 'relu',
                 layer_widths = [256, 128, 64],
                 batch_norm = False,
                 dropout = 0.0,
                 knn = 5, # Phate KNN
                 t = 'auto', # Phate t
                 n_landmark = 5000, # Phate n_landmark
                 verbose = False):
        super().__init__(ambient_dimension, latent_dimension, model_type)

        self.activation = activation
        self.layer_widths = layer_widths
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.knn = knn
        self.t = t
        self.n_landmark = n_landmark
        
        self.verbose = verbose

        self.encoder = None
        self.decoder = None
        
        self.input_dim = ambient_dimension
        self.latent_dim = latent_dimension

        # PHATE Operator
        self.phate_op = phate.PHATE(n_components=self.ambient_dimension, knn=self.knn, t=self.t, n_landmark=self.n_landmark)
    
    def fit(self, 
            X, 
            X_dist,
            train_mask,
            percent_test,
            data_name,
            mode,
            max_epochs,
            batch_size,
            lr,
            shuffle,
            componentwise_std,
            weight_decay,
            dist_mse_decay,
            dist_wieght=1.0,
            reconstr_weight=0.0,
            cycle_weight=0.0,
            cycle_dist_weight=0.0,
            monitor='validation/loss',
            patience=100,
            seed=2024,
            log_every_n_steps=100,
            accelerator='auto',
            train_from_scratch=True,
            model_save_path='./distance_matching/model.ckpt', # if None, train from scratch; else load from model_save_path
            ):
        
        seed_everything(seed)

        if train_mask is None:
            # Generate train_mask
            idxs = np.random.permutation(X.shape[0])
            split_idx = int(X.shape[0] * (1-percent_test))
            train_mask = np.zeros(X.shape[0], dtype=int)
            train_mask[idxs[:split_idx]] = 1
            train_mask = train_mask.astype(bool)

        # PHATE coordinates, gt distance matrix
        if X_dist is None:
            self.phate_coords = self.phate_op.fit_transform(X)
            self.diff_potential = self.phate_op.diff_potential
            phate_dist = scipy.spatial.distance.cdist(self.diff_potential, self.diff_potential) # [N, N]
        else:
            phate_dist = X_dist
            self.phate_coords = phate_coords
        dist_std = np.std(phate_dist.flatten())

        X = X[train_mask,:]
        phate_coords = self.phate_coords[train_mask,:]
        #colors = colors[train_mask]
        dist = phate_dist[train_mask,:][:,train_mask]

        train_loader, val_loader, mean, std = train_valid_loader_from_pc(
            X, # <---- Pointcloud
            dist, # <---- Distance matrix to match
            batch_size=batch_size,
            train_valid_split=0.8,
            shuffle=shuffle,
            seed=seed, 
            return_mean_std=True, 
            componentwise_std=componentwise_std)

        ''' Fit the model to the data X. '''
        cfg_dict = {
            'dimensions': {
                'data': self.input_dim,
                'latent': self.latent_dim
            },
            'encoder': {
                'layer_widths': self.layer_widths,
                'activation': self.activation,
                'batch_norm': self.batch_norm,
                'dropout': self.dropout,
            },
            'decoder': {
                'layer_widths': self.layer_widths[::-1],
                'activation': self.activation,
                'batch_norm': self.batch_norm,
                'dropout': self.dropout,
            },
            'loss': {
                'dist_mse_decay': dist_mse_decay,
                'weights': {
                    'dist': dist_wieght,
                    'reconstr': reconstr_weight,
                    'cycle': cycle_weight,
                    'cycle_dist': cycle_dist_weight
                }
            },
            'training': {
                'mode': mode,
                'max_epochs': max_epochs,
                'accelerator': accelerator,
                'lr': lr,
                'weight_decay': weight_decay,
                'dist_mse_decay': dist_mse_decay,
                'monitor': monitor,
                'patience': patience,
                'log_every_n_steps': log_every_n_steps,
            },
            'logger': {
                'use_wandb': False,
            },
            'path': {
                'root': os.path.dirname(model_save_path),
                'model': os.path.basename(model_save_path),
                'log': 'mylogs'
            },
        }
        cfg = OmegaConf.create(cfg_dict)
        print('Fitting model ...', cfg)

        model = Autoencoder(cfg, preprocessor=Preprocessor(mean, std, dist_std))
        
        device_av = "cuda" if torch.cuda.is_available() else "cpu"
        if accelerator is None or accelerator == 'auto':
            device = device_av
        else:
            device = accelerator
        self.device = device

        if train_from_scratch is False:
            if cfg.training.mode == 'encoder':
                encoder = Encoder.load_from_checkpoint(model_save_path + '.ckpt')
                self.encoder = encoder
                self.decoder = model.decoder # a random decoder
            elif cfg.training.mode == 'decoder':
                decoder = Decoder.load_from_checkpoint(model_save_path + '.ckpt')
                self.decoder = decoder
                self.encoder = model.encoder # a random encoder
            else:
                model = Autoencoder.load_from_checkpoint(model_save_path + '.ckpt')
                self.encoder = model.encoder
                self.decoder = model.decoder
            
            print(f'Loaded encoder from {model_save_path}, skipping encoder training ...')
            print(f'Loaded decoder from {model_save_path}, skipping decoder training ...')

            return
 
        os.makedirs(cfg.path.root, exist_ok=True)

        if cfg.logger.use_wandb:
            logger = WandbLogger()
        else:
            logger = TensorBoardLogger(save_dir=os.path.join(cfg.path.root, cfg.path.log))

        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.path.root,  # Save checkpoints in wandb directory
            filename=cfg.path.model,
            save_top_k=1,
            monitor=cfg.training.monitor,  # Model selection based on validation loss
            mode='min',  # Minimize validation loss

        )

        if cfg.training.mode == 'separate':
            cfg.training.mode = 'encoder'
            train_model(cfg, model.encoder, train_loader, val_loader, logger, checkpoint_callback)
            model.link_encoder()
            cfg.training.mode = 'decoder'
            train_model(cfg, model.decoder, train_loader, val_loader, logger, checkpoint_callback)
        elif cfg.training.mode == 'end2end':
            train_model(cfg, model, train_loader, val_loader, logger, checkpoint_callback)
        elif cfg.training.mode == 'encoder':
            train_model(cfg, model.encoder, train_loader, val_loader, logger, checkpoint_callback)
        elif cfg.training.mode == 'decoder':
            train_model(cfg, model.decoder, train_loader, val_loader, logger, checkpoint_callback)
        else:
            raise ValueError('Invalid training mode')            

        if cfg.training.mode == 'encoder':
            model.encoder.load_from_checkpoint(model_save_path + '.ckpt')
        elif cfg.training.mode == 'decoder':
            model.decoder.load_from_checkpoint(model_save_path + '.ckpt')
        else:
            model.load_from_checkpoint(model_save_path + '.ckpt')

        self.encoder = model.encoder
        self.decoder = model.decoder

        print('Done fitting model.')

    
    def encode(self, X):
        ''' Encode input data X to latent space. '''
        if self.encoder is None:
            raise ValueError('Encoder not trained yet. Please train the model first.')
        X = X.to(self.device)

        self.encoder.eval()
        Z = self.encoder(X)
        
        return Z
    
    def decode(self, Z):
        ''' Decode latent space Z to ambient space. '''
        if self.decoder is None:
            raise ValueError('Decoder not trained yet. Please train the model first.')
        Z = Z.to(self.device)
        self.decoder.eval()
        X_hat = self.decoder(Z)
        
        return X_hat


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_name', type=str, default='swiss_roll')
    argparser.add_argument('--mode', type=str, default='encoder')
    argparser.add_argument('--from_scratch', action='store_true')

    args = argparser.parse_args()
    data_name = args.data_name
    mode = args.mode
    from_scratch = args.from_scratch

    save_folder = f'./{data_name}/dmatch_{mode}'

    # Data
    data_name = 'swiss_roll'
    if data_name == 'swiss_roll':
        gt_X, X, _ = sklearn_swiss_roll(n_samples=1000, noise=0.0)
        colors = None
    elif data_name == 'hemisphere':
        gt_X, X, _ = hemisphere_data(n_samples=1000, noise=0.0)
        colors = None
    
    if from_scratch:
        os.system(f'rm -rf {save_folder}')
    os.makedirs(save_folder, exist_ok=True)

    model_hypers = {
        'ambient_dimension': 3,
        'latent_dimension': 3,
        'model_type': 'distance',
        'activation': 'relu',
        'layer_widths': [256, 128, 64],
        'knn': 5,
        't': 'auto',
        'verbose': False
    }
    training_hypers = {
        'data_name': 'randomtest',
        'mode': mode, # 'encoder', 'decoder', 'end2end', 'separate
        'max_epochs': 10,
        'batch_size': 64,
        'lr': 1e-3,
        'shuffle': True,
        'componentwise_std': False,
        'weight_decay': 1e-5,
        'dist_mse_decay': 0,
        'dist_wieght': 0.9,
        'reconstr_weight': 0.1,
        'cycle_weight': 0.1,
        'cycle_dist_weight': 0.1,
        'monitor': 'validation/loss',
        'patience': 100,
        'seed': 2024,
        'log_every_n_steps': 100,
        'accelerator': 'auto',
        'train_from_scratch': from_scratch,
        'model_save_path': f'{save_folder}/model'
    }

    print('Fitting on X: ', gt_X.shape, X.shape)
    model = DistanceMatching(**model_hypers)
    model.fit(X, X_dist=None, train_mask=None, percent_test=0.2, **training_hypers)

    X = torch.tensor(X, dtype=torch.float32)
    Z = model.encode(X)
    print('Encoded Z:', Z.shape)
    X_hat = model.decode(Z)
    print('Decoded X:', X_hat.shape)
    phate_coords = model.phate_coords
    print('PHATE Coords:', phate_coords.shape)

    # Plot
    fig = plt.figure(figsize=(24, 8))
    ax = fig.add_subplot(141, projection='3d')
    scprep.plot.scatter3d(X.detach().cpu().numpy(), c=colors, ax=ax, title='X')

    ax = fig.add_subplot(142, projection='3d')
    scprep.plot.scatter3d(Z.detach().cpu().numpy(), c=colors, ax=ax, title='Z')

    ax = fig.add_subplot(143, projection='3d')
    scprep.plot.scatter3d(phate_coords, c=colors, ax=ax, title='Phate')

    ax = fig.add_subplot(144, projection='3d')
    scprep.plot.scatter3d(X_hat.detach().cpu().numpy(), c=colors, ax=ax, title='X_hat')

    plt.savefig(f'{save_folder}/plot.png')
