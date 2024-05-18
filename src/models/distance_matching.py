import sys
import os

import numpy as np
import torch
import phate
import scipy
import matplotlib.pyplot as plt
import scprep
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from omegaconf import OmegaConf

sys.path.append('../../src/')
from data import train_valid_loader_from_pc
from model2 import Encoder, Decoder, Preprocessor, Autoencoder, WDiscriminator
from models.unified_model import GeometricAE
from off_manifolder import offmanifolder_maker
from geodesic import GeodesicBridgeOverfit, GeodesicBridge
from data_convert import convert_data
from negative_sampling import add_negative_samples
from data_script import hemisphere_data, sklearn_swiss_roll
from utils.seed import seed_everything


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
                 verbose = False):
        super().__init__(ambient_dimension, latent_dimension, model_type)

        self.activation = activation
        self.layer_widths = layer_widths
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.knn = knn
        self.t = t
        
        self.verbose = verbose

        self.encoder = None
        self.decoder = None
        
        self.input_dim = ambient_dimension
        self.latent_dim = latent_dimension

        # PHATE Operator
        self.phate_op = phate.PHATE(n_components=self.ambient_dimension, knn=self.knn, t=self.t)
    
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
            clip=True,
            spectral_norm=True,
            clamp=1.0,
            pos1=1.0,
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

        # Subset on train mask
        X = X[train_mask,:]
        phate_coords = self.phate_coords[train_mask,:]
        #colors = colors[train_mask]
        dist = phate_dist[train_mask,:][:,train_mask]

        train_loader, val_loader, mean, std = train_valid_loader_from_pc(
            X, # <---- Pointcloud
            dist, # <---- Distance matrix to match
            mask_x=None, # <---- Mask for pointcloud, 1 for pos, 0 for neg
            mask_d=None, # <---- Mask for distance matrix
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
                    'cycle_dist': cycle_dist_weight,
                    'wgan': 1.0,
                    'pos1': pos1,
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
                'shuffle': shuffle,
                'batch_size': batch_size,
                'train_valid_split': 0.8,
                'clip': clip,
                'clamp': clamp,
                'spectral_norm': spectral_norm,
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
        self.cfg = cfg
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
            elif cfg.training.mode == 'end2end':
                model = Autoencoder.load_from_checkpoint(model_save_path + '.ckpt')
                self.encoder = model.encoder
                self.decoder = model.decoder
            elif cfg.training.mode == 'separate':
                encoder = Encoder.load_from_checkpoint(model_save_path + '_encoder.ckpt')
                # filter out unnecessary keys
                state_dict = torch.load(model_save_path + '_decoder.ckpt')['state_dict']
                state_dict = {k: v for k, v in state_dict.items() if 'encoder' not in k}
                model.decoder.load_state_dict(state_dict)
                self.encoder = encoder
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
            encoder_checkpoint_callback = ModelCheckpoint(
                dirpath=cfg.path.root,  # Save checkpoints in wandb directory
                filename=f'{cfg.path.model}_encoder',
                save_top_k=1,
                monitor=cfg.training.monitor,  # Model selection based on validation loss
                mode='min',  # Minimize validation loss
            )
            print('=====Training encoder ..., save at ', encoder_checkpoint_callback.filename)
            train_model(cfg, model.encoder, train_loader, val_loader, logger, encoder_checkpoint_callback)
            model.link_encoder()

            cfg.training.mode = 'decoder'
            decoder_checkpoint_callback = ModelCheckpoint(
                dirpath=cfg.path.root,  # Save checkpoints in wandb directory
                filename=f'{cfg.path.model}_decoder',
                save_top_k=1,
                monitor=cfg.training.monitor,  # Model selection based on validation loss
                mode='min',  # Minimize validation loss
            )
            train_model(cfg, model.decoder, train_loader, val_loader, logger, decoder_checkpoint_callback)
            print('=====Training decoder ..., save at ', decoder_checkpoint_callback.filename)

            cfg.training.mode = 'separate'
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
        elif cfg.training.mode == 'end2end':
            model.load_from_checkpoint(model_save_path + '.ckpt')
        elif cfg.training.mode == 'separate':
            state_dict = torch.load(model_save_path + '_encoder.ckpt')['state_dict']
            print('Encoder state_dict:', state_dict.keys())
            model.encoder.load_from_checkpoint(model_save_path + '_encoder.ckpt')

            state_dict = torch.load(model_save_path + '_decoder.ckpt')['state_dict']
            print('Decoder state_dict:', state_dict.keys())
            # model.decoder.load_from_checkpoint(model_save_path + '_decoder.ckpt')
            # filter out unnecessary keys
            #state_dict = {k: v for k, v in state_dict.items() if 'encoder' not in k}
            model.decoder.load_state_dict(state_dict)

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
    
    def fit_wdiscriminator(self, og_X, seed):
        '''
        Fit a Wasserstein discriminator on X, negative Xs.

        '''
        print('Fitting Wasserstein Discriminator ...')

        # Convert X to dict format for negative sampling
        og_data_dict = convert_data(og_X)

        # Negative sampling s.t. data_dict['data'] is X + noise
        noise_type = 'hi-freq-no-add'
        noise_level = 1.1
        mask_dist = False
        data_dict = add_negative_samples(og_data_dict.copy(), subset_rate=1., noise_rate=noise_level, seed=seed, 
                                         noise=noise_type, mask_dists=mask_dist, shell=True)

        X = data_dict['data'].astype(np.float32)
        dist = data_dict['dist'].astype(np.float32)
        colors = data_dict.get('colors', None)
        dist_std = np.std(dist.flatten())
        mask_x = data_dict.get('mask_x', None)
        mask_d = data_dict.get('mask_d', None)

        # Subset on train mask
        train_mask = data_dict['is_train'].astype(bool)
        X = X[train_mask,:]
        dist = dist[train_mask,:][:,train_mask]
        colors = colors[train_mask]
        if mask_x is not None:
            mask_x = mask_x.astype(bool)
            mask_x = mask_x[train_mask]
            if len(mask_x.shape) == 1:
                mask_x = mask_x.reshape(-1,1)
        if mask_d is not None:
            mask_d = mask_d[train_mask,:][:,train_mask]
        
        trainloader, valloader, mean, std = train_valid_loader_from_pc(
            X, # <---- Pointcloud
            dist, # <---- Distance matrix to match
            mask_x=mask_x,
            mask_d=mask_d,
            batch_size=self.cfg.training.batch_size,
            train_valid_split=self.cfg.training.train_valid_split,
            shuffle=self.cfg.training.shuffle,
            seed=seed, 
            return_mean_std=True, 
            componentwise_std=False)
        
        self.mask_x = mask_x
        self.wgan_x = X # train data for w-critic
        
        preprocessor = Preprocessor(mean=mean, std=std, dist_std=dist_std)
        wd_cfg = self.cfg.copy()
        wd_cfg.encoder.dropout = 0.5
        wd_cfg.encoder.batch_norm = True
        wd_cfg.encoder.spectral_norm = True
        wd_cfg.training.weight_decay = 1e-4
        
        # Fit Wasserstein Discriminator.
        wd_model = WDiscriminator(wd_cfg, preprocessor=preprocessor)

        if self.cfg.logger.use_wandb:
            logger = WandbLogger()
        else:
            logger = TensorBoardLogger(save_dir=os.path.join(self.cfg.path.root, self.cfg.path.log))
        
        early_stopping = EarlyStopping('validation/loss', patience=self.cfg.training.patience)

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.cfg.path.root,  # Save checkpoints in wandb directory
            filename='w_critc',
            save_top_k=1,
            monitor='validation/loss',  # Model selection based on validation loss
            mode='min',  # Minimize validation loss
        )

        trainer = pl.Trainer(logger=logger,
                             max_epochs=self.cfg.training.max_epochs,
                             accelerator=self.cfg.training.accelerator,
                             callbacks=[early_stopping, checkpoint_callback],
                             log_every_n_steps=self.cfg.training.log_every_n_steps,
                             )
        trainer.fit(wd_model, trainloader, valloader)

        self.w_discriminator = wd_model
        print('Done fitting Wasserstein Discriminator.')
        

    def fit_geodesic(self, X, starts, ends, ts):
        '''
        Fit a geodesic between start and end in ambient space.
        starts, ends: torch.tensor of shape [N, d]
        ts: torch.tensor of shape [N, 1] time points where to evaluate the geodesic.

        Returns:
            geodesic: torch.tensor of shape [N, len(ts), d] geodesic points.
        '''

        wd = self.w_discriminator
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=wd.device)

        with torch.no_grad():
            wd.eval()
            probab = wd(X).flatten()
        
        start_batch = torch.tensor(starts, dtype=X.dtype, device=X.device)
        end_batch = torch.tensor(ends, dtype=X.dtype, device=X.device)
        ids = torch.zeros((start_batch.size(0),1))

        dataset = TensorDataset(start_batch, end_batch, ids)
        dataloader = DataLoader(dataset, batch_size=len(Z), shuffle=True)

        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in wd.parameters():
            param.requires_grad = False
        
        enc_func = lambda x: model.encoder(x)
        disc_func = lambda x: (wd(x).flatten()-probab.min())/(probab.max()-probab.min())

        ofm = offmanifolder_maker(enc_func, disc_func, disc_factor=0.5, max_prob=probab.max())
        gbmodel = GeodesicBridgeOverfit(
            func=ofm,
            input_dim=X.size(1), 
            hidden_dim=64, 
            scale_factor=1, 
            symmetric=True, 
            num_layers=3, 
            n_tsteps=len(ts),
            lr=1e-3, 
            weight_decay=1e-3,
            discriminator_weight=0.,
            discriminator_func_for_grad_weight=0.,
            id_dim=1,
            id_emb_dim=1,
            density_weight=0.,
            length_weight=1.,
        )

        gbmodel.train() # Set to train mode
        if self.cfg.logger.use_wandb:
            logger = WandbLogger()
        else:
            logger = TensorBoardLogger(save_dir=os.path.join(self.cfg.path.root, self.cfg.path.log))

        early_stopping = EarlyStopping('loss', patience=50)
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.cfg.path.root,  # Save checkpoints in wandb directory
            filename='geodesic',
            save_top_k=1,
            monitor='loss',  # Model selection based on validation loss
            mode='min',  # Minimize validation loss
        )
        trainer = pl.Trainer(logger=logger,
                             max_epochs=self.cfg.training.max_epochs,
                             accelerator=self.cfg.training.accelerator,
                             callbacks=[early_stopping, checkpoint_callback],
                             log_every_n_steps=self.cfg.training.log_every_n_steps,
                             )
        trainer.fit(gbmodel, dataloader)

        self.geo_model = gbmodel
        print('Done fitting geodesic.')


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_name', type=str, default='swiss_roll')
    argparser.add_argument('--mode', type=str, default='end2end')
    argparser.add_argument('--epoch', type=int, default=1)
    argparser.add_argument('--from_scratch', action='store_true')
    argparser.add_argument('--seed', type=int, default=2024)

    args = argparser.parse_args()
    data_name = args.data_name
    mode = args.mode
    max_epochs = args.epoch
    from_scratch = args.from_scratch
    seed = args.seed

    save_folder = f'./{data_name}/dmatch_{mode}'

    # Data
    if data_name == 'swiss_roll':
        gt_X, X, _ = sklearn_swiss_roll(n_samples=1000, noise=0.0)
        colors = None
    elif data_name == 'hemisphere':
        gt_X, X, _ = hemisphere_data(n_samples=3000, noise=0.0)
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
        'max_epochs': max_epochs,
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
        'seed': seed,
        'log_every_n_steps': 100,
        'accelerator': 'auto',
        'train_from_scratch': from_scratch,
        'model_save_path': f'{save_folder}/model',
        # for Wasserstein Discriminator
        'spectral_norm': True,
        'clip': True,
        'clamp': 0.1,
        'pos1': 1.0,
    }

    print('Fitting on X: ', gt_X.shape, X.shape)
    model = DistanceMatching(**model_hypers)
    model.fit(X, X_dist=None, train_mask=None, percent_test=0.2, **training_hypers)

    # Fit Wasserstein Discriminator
    model.fit_wdiscriminator(X, seed=seed)

    X = torch.tensor(X, dtype=torch.float32)
    Z = model.encode(X)
    print('Encoded Z:', Z.shape)
    X_hat = model.decode(Z)
    print('Decoded X:', X_hat.shape)
    phate_coords = model.phate_coords
    print('PHATE Coords:', phate_coords.shape)

    # Plotly on wasserstein discriminator prob on X, show bar color
    wd = model.w_discriminator
    mask_x = model.mask_x.flatten()
    wd_x = model.wgan_x
    # wd.eval()
    probab = wd(torch.Tensor(wd_x)).flatten().detach().cpu().numpy()
    print('WD X:', wd_x.shape, mask_x.shape, probab.shape)
    print(mask_x.dtype, probab.dtype)

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=wd_x[:,0][mask_x], y=wd_x[:,1][mask_x], z=wd_x[:,2][mask_x], mode='markers', marker=dict(size=3, color=probab[mask_x], colorscale='Viridis')))
    fig.add_trace(go.Scatter3d(x=wd_x[:,0][~mask_x], y=wd_x[:,1][~mask_x], z=wd_x[:,2][~mask_x], mode='markers', marker=dict(size=3, color=probab[~mask_x], colorscale='Viridis')))    
    fig.write_html(f'{save_folder}/w_critic.html')

    # Fit Geodesic
    num_endpoints = 32
    starts = X[np.random.randint(0, X.shape[0], num_endpoints), :]
    ends = X[np.random.randint(0, X.shape[0], num_endpoints), :]
    ts = torch.linspace(0, 1, 100).reshape(-1, 1)

    model.fit_geodesic(X, starts, ends, ts)
    geo_model = model.geo_model

    ids = torch.zeros((starts.shape[0],1))
    pred_geodesic = geo_model(starts, ends, ts, ids).detach().cpu().numpy()
    print('Predicated Geodesic:', pred_geodesic.shape)
    
    # Plot starts, ends, geodesic
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2], mode='markers', marker=dict(size=3, color='gray')))
    fig.add_trace(go.Scatter3d(x=starts[:,0], y=starts[:,1], z=starts[:,2], mode='markers', marker=dict(size=3, color='red')))
    fig.add_trace(go.Scatter3d(x=ends[:,0], y=ends[:,1], z=ends[:,2], mode='markers', marker=dict(size=3, color='blue')))
    for i in range(num_endpoints):
        fig.add_trace(go.Scatter3d(x=pred_geodesic[:,i,0], y=pred_geodesic[:,i,1], z=pred_geodesic[:,i,2], mode='lines', line=dict(width=2)))
    fig.write_html(f'{save_folder}/geodesic.html')

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
