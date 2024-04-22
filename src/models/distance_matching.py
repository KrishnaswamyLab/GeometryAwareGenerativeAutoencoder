import sys
import os
import numpy as np
import torch
import phate
import scipy

sys.path.append('../../src/')
from data import train_valid_loader_from_pc
from model2 import Encoder, Decoder, Preprocessor, Autoencoder
# from train import train_model
from models.unified_model import GeometricAE

from utils.seed import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

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
        self.phate_coords = self.phate_op.fit_transform(X)
        self.diff_potential = self.phate_op.diff_potential
        phate_dist = scipy.spatial.distance.cdist(self.diff_potential, self.diff_potential) # [N, N]
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
                    'dist': 0.9,
                    'reconstr': 0.1,
                    'cycle': 0.0,
                    'cycle_dist': 0.0,
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
            model.load_from_checkpoint(model_save_path + '.ckpt')
            print(f'Loaded encoder from {model_save_path}, skipping encoder training ...')
            print(f'Loaded decoder from {model_save_path}, skipping decoder training ...')

            self.encoder = model.encoder
            self.decoder = model.decoder

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
        elif cfg.training.mode == 'end2end': # encoder-only, decoder-only, or end-to-end
            train_model(cfg, model, train_loader, val_loader, logger, checkpoint_callback)
        elif cfg.training.mode == 'encoder':
            train_model(cfg, model.encoder, train_loader, val_loader, logger, checkpoint_callback)
        elif cfg.training.mode == 'decoder':
            train_model(cfg, model.decoder, train_loader, val_loader, logger, checkpoint_callback)
        else:
            raise ValueError('Invalid training mode')            
        ckpt = torch.load(model_save_path + '.ckpt')
        print(ckpt['state_dict'].keys())

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
        # with torch.no_grad(): # Need gradients for pullback metrics
            # X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Z = self.encoder(X)
        
        return Z #.detach().cpu().numpy()
    
    def decode(self, Z):
        ''' Decode latent space Z to ambient space. '''
        if self.decoder is None:
            raise ValueError('Decoder not trained yet. Please train the model first.')
        Z = Z.to(self.device)
        self.decoder.eval()
        # with torch.no_grad():
        #     Z = torch.tensor(Z, dtype=torch.float32).to(self.device)
        X_hat = self.decoder(Z)
        
        return X_hat #.detach().cpu().numpy()
    


if __name__ == "__main__":
    mode = 'encoder'
    model_hypers = {
        'ambient_dimension': 10,
        'latent_dimension': 2,
        'model_type': 'distance',
        'activation': 'relu',
        'layer_widths': [256, 128, 64],
        'knn': 5,
        't': 'auto',
        'n_landmark': 5000,
        'verbose': False
    }
    training_hypers = {
        'data_name': 'randomtest',
        'mode': mode, # 'encoder', 'decoder', 'end2end', 'separate
        'max_epochs': 1,
        'batch_size': 64,
        'lr': 1e-3,
        'shuffle': True,
        'componentwise_std': False,
        'weight_decay': 1e-5,
        'dist_mse_decay': 1e-5,
        'monitor': 'validation/loss',
        'patience': 100,
        'seed': 2024,
        'log_every_n_steps': 100,
        'accelerator': 'auto',
        'train_from_scratch': True,
        'model_save_path': f'./distance_matching_{mode}/model'
    }
    # Test AffinityMatching model
    X = np.random.randn(100, 10) # 3000 samples, 10 features
    model = DistanceMatching(**model_hypers)
    model.fit(X, train_mask=None, percent_test=0.3, **training_hypers)

    X = torch.tensor(X, dtype=torch.float32)
    Z = model.encode(X)
    print('Encoded Z:', Z.shape)
    X_hat = model.decode(Z)
    print('Decoded X:', X_hat.shape)

    # Pullback metrics
    # metric = model.encoder_pullback(X)
    # print('X: ', X.shape, 'metric: ', metric.shape)

    #geodesic pullback
    T = 5
    X_tb = np.random.randn(16, T, 10)
    X_tb = torch.tensor(X_tb, dtype=torch.float32)
    metric_tb = model.geodesic_encoder_pullback(X_tb)
    print('X_tb: ', X_tb.shape, 'metric_tb: ', metric_tb.shape)
