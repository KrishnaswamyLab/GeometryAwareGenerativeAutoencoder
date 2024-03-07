import os
import pickle
import matplotlib.pyplot as plt
import wandb
import hydra
import yaml
import numpy as np
import pandas as pd
import torch
import scipy.sparse
from scipy.spatial.distance import pdist, squareform
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from data import train_valid_loader_from_pc, dataloader_from_pc
from transformations import LogTransform, NonTransform, StandardScaler, \
    MinMaxScaler, PowerTransformer, KernelTransform
from model import AEDist, VAEDist
from metrics import distance_distortion, mAP
from procrustes import Procrustes

def to_dense_array(X):
    if scipy.sparse.issparse(X):  # Check if X is a sparse matrix
        return X.toarray()
    elif isinstance(X, np.ndarray):  # Check if X is already a numpy array
        return X
    elif isinstance(X, pd.DataFrame):  # Check if X is a pandas DataFrame
        return X.values  # or X.to_numpy()
    else:
        raise TypeError("Input is neither a sparse matrix, a numpy array, nor a pandas DataFrame")

@hydra.main(version_base=None, config_path='../conf', config_name='config')
def main(cfg: DictConfig):
    if cfg.logger.use_wandb:
        config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        run = wandb.init(
            entity=cfg.logger.entity,
            project=cfg.logger.project,
            tags=cfg.logger.tags,
            reinit=True,
            config=config,
            settings=wandb.Settings(start_method="thread"),
        )
    
    model, trainloader, valloader, X, phate_coords, colors, dist = prep_data_model(cfg)

    early_stopping = EarlyStopping(cfg.training.monitor, patience=cfg.training.patience)
    if cfg.logger.use_wandb:
        logger = WandbLogger()
        checkpoint_callback = ModelCheckpoint(
            dirpath=wandb.run.dir,  # Save checkpoints in wandb directory
            save_top_k=1,  # Save the top 1 model
            monitor='val_loss',  # Model selection based on validation loss
            mode='min'  # Minimize validation loss
        )
    else:
        logger = TensorBoardLogger(save_dir=os.path.join(cfg.path.root, cfg.path.log))
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.path.root,  # Save checkpoints in wandb directory
            filename=cfg.path.model,
            save_top_k=1,
            monitor='val_loss',  # Model selection based on validation loss
            mode='min'  # Minimize validation loss
        )

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

    X_tensor = torch.from_numpy(X).float()
    x_hat, emb_z = model(X_tensor)
    x_hat = x_hat.cpu().detach().numpy()
    emb_z = emb_z.cpu().detach().numpy()

    procrustes = Procrustes()
    pc_s, z, disparity = procrustes.fit_transform(phate_coords, emb_z)
    if cfg.path.save:
        with open(os.path.join(cfg.path.root, f'{cfg.path.procrustes}.pkl'), 'wb') as file:
            pickle.dump(procrustes, file)

    if cfg.logger.use_wandb:
        wandb.log({'procrustes_disparity_latent': disparity})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.scatter(phate_coords[:,0], phate_coords[:,1], c=colors, s=1, cmap='Spectral')
    ax1.set_title('PHATE')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.scatter(z[:,0], z[:,1], c=colors, s=1, cmap='Spectral')
    ax2.set_title('Latent Space')
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.suptitle('Comparison of PHATE and Latent Space')
    
    if cfg.path.save:
        plotdir = os.path.join(cfg.path.root, cfg.path.plots)
        os.makedirs(plotdir, exist_ok=True)
        plt.savefig(f'{plotdir}/comparison_latent.pdf', dpi=300)

    if cfg.logger.use_wandb:
        wandb.log({'Comparison Plot Latent': plt})

    procrustes = Procrustes()
    xo, xh, disparity = procrustes.fit_transform(X, x_hat)
    if cfg.logger.use_wandb:
        wandb.log({'procrustes_disparity_reconstruction': disparity})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.scatter(xo[:,0], xo[:,1], c=colors, s=1, cmap='Spectral')
    ax1.set_title('PHATE')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.scatter(xh[:,0], xh[:,1], c=colors, s=1, cmap='Spectral')
    ax2.set_title('Latent Space')
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.suptitle('Comparison of PHATE and Latent Space')
    
    if cfg.path.save:
        plotdir = os.path.join(cfg.path.root, cfg.path.plots)
        os.makedirs(plotdir, exist_ok=True)
        plt.savefig(f'{plotdir}/comparison_reconstr.pdf', dpi=300)

    if cfg.logger.use_wandb:
        wandb.log({'Comparison Plot Reconstruction': plt})
        dist_distort = distance_distortion(dist, squareform(pdist(emb_z)))
        wandb.log({'distance_distortion': dist_distort})
        # TODO mAP score needs input graph.
        run.finish()


def load_data(cfg, load_all=False):
    # if load_all:
    #     data_path = os.path.join(cfg.data.root, cfg.data.name + "_all" + cfg.data.filetype)
    # else:
    #     data_path = os.path.join(cfg.data.root, cfg.data.name + cfg.data.filetype)
    data_path = os.path.join(cfg.data.root, cfg.data.name + cfg.data.filetype)
    data = np.load(data_path, allow_pickle=True)
    # sanity check the data is not empty
    assert 'data' in data.files and 'phate' in data.files and 'colors' in data.files \
        and 'dist' in data.files, "Some required files are missing in the 'data' variable."
    X = data['data']
    phate_coords = data['phate']
    colors = data['colors']
    dist = data['dist']
    train_mask = data['is_train']
    if not load_all:
        X = X[train_mask,:]
        phate_coords = phate_coords[train_mask,:]
        colors = colors[train_mask]
        dist = dist[train_mask,:][:,train_mask]
    assert X.shape[0] == phate_coords.shape[0] == colors.shape[0] == dist.shape[0], \
        "The number of cells in the data, phate, and colors variables do not match."

    if cfg.training.match_potential:
        phate_D = dist
    else:
        phate_D = squareform(pdist(phate_coords))
    
    ##### [TMP FIX] to be compatible with old runs that don't have the data.kernel field
    if cfg.data.preprocess == 'kernel': 
        preprocessor_dict = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'power': PowerTransformer(),
            'log': LogTransform(),
            'none': NonTransform(),
            'kernel': KernelTransform(
                cfg.data.kernel.type, 
                cfg.data.kernel.sigma, 
                cfg.data.kernel.epsilon, 
                cfg.data.kernel.alpha,
                cfg.data.kernel.use_std
                )
        }
    else:
        preprocessor_dict = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'power': PowerTransformer(),
            'log': LogTransform(),
            'none': NonTransform(),
        }
    
    pp = preprocessor_dict[cfg.data.preprocess]
    shapes = phate_D.shape
    phate_D = pp.fit_transform(phate_D.reshape(-1,1)).reshape(shapes)
    if load_all:
        # DEPRECATED
        allloader = dataloader_from_pc(
        X, # <---- Pointcloud
        phate_D, # <---- Distance matrix to match
        batch_size=X.shape[0],
        shuffle=False,)
        valloader = None
    else:
        trainloader, valloader, mean, std = train_valid_loader_from_pc(
            X, # <---- Pointcloud
            phate_D, # <---- Distance matrix to match
            batch_size=cfg.training.batch_size,
            train_valid_split=cfg.training.train_valid_split,
            shuffle=cfg.training.shuffle,
            seed=cfg.training.seed,return_mean_std=True)
    
    return trainloader, valloader, X, phate_coords, colors, dist, pp, mean, std

def make_model(cfg, dim, emb_dim, pp, dist_std, mean, std, from_checkpoint=False, checkpoint_path=None):
    if from_checkpoint:
        return AEDist.load_from_checkpoint(checkpoint_path)
    if 'emb_dim' in cfg.model:
        emb_dim = cfg.model.emb_dim
    activation_dict = {
        'relu': torch.nn.ReLU(),
        'leaky_relu': torch.nn.LeakyReLU(),
        'sigmoid': torch.nn.Sigmoid()
    }

    if 'use_dist_mse_decay' in cfg.model: # for compatibility with old runs
        use_dist_mse_decay = cfg.model.use_dist_mse_decay
        dist_mse_decay = cfg.model.dist_mse_decay / dist_std
    else:
        use_dist_mse_decay = False
        dist_mse_decay = 0.0
    if 'weight_decay' not in cfg.model:
        cfg.model.weight_decay = 0.0 # for compatibility with old runs
    if 'dropout' not in cfg.model:
        cfg.model.dropout = 0.0 # for compatibility with old runs
    if 'batch_norm' not in cfg.model:
        cfg.model.batch_norm = False
    if 'cycle_weight' not in cfg.model:
        cfg.model.cycle_weight = 0.0
    if 'cycle_dist_weight' not in cfg.model:
        cfg.model.cycle_dist_weight = 0.0
    if 'normalize' not in cfg.model:
        cfg.model.normalize = False
    if not cfg.model.normalize:
        mean = None
        std = None
    activation_fn = activation_dict[cfg.model.activation]
    if cfg.model.type == 'ae':
        # if from_checkpoint:
        #     model = AEDist.load_from_checkpoint(
        #         checkpoint_path=checkpoint_path,
        #         dim=dim,
        #         emb_dim=emb_dim,
        #         layer_widths=cfg.model.layer_widths,
        #         activation_fn=activation_fn,
        #         dist_reconstr_weights=cfg.model.dist_reconstr_weights,
        #         pp=pp,
        #         lr=cfg.model.lr,
        #         weight_decay=cfg.model.weight_decay,
        #         batch_norm=cfg.model.batch_norm,
        #         dist_recon_topk_coords=cfg.model.dist_recon_topk_coords,
        #         use_dist_mse_decay=use_dist_mse_decay,
        #         dist_mse_decay=dist_mse_decay,
        #         dropout=cfg.model.dropout,
        #         cycle_weight=cfg.model.cycle_weight,
        #         cycle_dist_weight=cfg.model.cycle_dist_weight,
        #         # mean=mean,  # they are saved in the checkpoint.
        #         # std=std,
        #     )
        # else:
        model = AEDist(
            dim=dim,
            emb_dim=emb_dim,
            layer_widths=cfg.model.layer_widths,
            activation_fn=activation_fn,
            dist_reconstr_weights=cfg.model.dist_reconstr_weights,
            pp=pp,
            lr=cfg.model.lr,
            weight_decay=cfg.model.weight_decay,
            batch_norm=cfg.model.batch_norm,
            dist_recon_topk_coords=cfg.model.dist_recon_topk_coords,
            use_dist_mse_decay=use_dist_mse_decay,
            dist_mse_decay=dist_mse_decay,
            dropout=cfg.model.dropout,
            cycle_weight=cfg.model.cycle_weight,
            cycle_dist_weight=cfg.model.cycle_dist_weight,
            mean=mean,
            std=std,
        )
    elif cfg.model.type == 'vae':
        # DEPRECATED
        NotImplemented
        # TODO add dist_mse_decay to VAE?
        if from_checkpoint:
            model = VAEDist.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                dim=dim,
                emb_dim=emb_dim,
                layer_widths=cfg.model.layer_widths,
                activation_fn=activation_fn,
                dist_reconstr_weights=cfg.model.dist_reconstr_weights,
                kl_weight=cfg.model.kl_weight,
                pp=pp,
                lr=cfg.model.lr,
                dist_recon_topk_coords=cfg.model.dist_recon_topk_coords,
            )
        else:
            model = VAEDist(
                dim=dim,
                emb_dim=emb_dim,
                layer_widths=cfg.model.layer_widths,
                activation_fn=activation_fn,
                dist_reconstr_weights=cfg.model.dist_reconstr_weights,
                kl_weight=cfg.model.kl_weight,
                pp=pp,
                lr=cfg.model.lr,
                dist_recon_topk_coords=cfg.model.dist_recon_topk_coords,
            )
    else:
        raise NotImplementedError(f"Model type {cfg.model.type} not implemented.")

    return model

def prep_data_model(cfg):
    trainloader, valloader, X, phate_coords, colors, dist, pp, mean, std = load_data(cfg)
    dist_std = np.std(dist.flatten())
    model = make_model(cfg, X.shape[1], phate_coords.shape[1], pp, dist_std, mean, std)

    return model, trainloader, valloader, X, phate_coords, colors, dist

if __name__ == "__main__":
    main()