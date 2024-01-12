import sys
from data import train_and_testloader_from_pc
from model import AEDist
import numpy as np
import scipy.sparse
import pandas as pd
import torch
import phate
# from heatgeo.embedding import HeatGeo
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from procrustes import Procrustes
import pickle
import scanpy as sc
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import hydra
import os
from omegaconf import DictConfig, OmegaConf


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

    if cfg.data.file_type == 'h5ad':
        adata = sc.read_h5ad(cfg.data.datapath)
        X = to_dense_array(adata.X[:,:])
        if not cfg.data.require_phate:
            phate_coords = adata.obsm[cfg.data.adata_phate_name]
            emb_dim = phate_coords.shape[1]
    elif cfg.data.file_type == 'npy':
        X = np.load(cfg.data.datapath)
        if not cfg.data.require_phate:
            phate_coords = np.load(cfg.data.phatepath)
            emb_dim = phate_coords.shape[1]
    else:
        raise ValueError('Unknown file type')
    if cfg.data.require_phate:
        emb_dim = cfg.data.phate_dim
        phate_op = phate.PHATE(
            random_state=cfg.phate.random_state,
            n_components=emb_dim,
            knn=cfg.phate.knn,
            decay=cfg.phate.decay,
            t=cfg.phate.t,
            n_jobs=cfg.phate.n_jobs,
        )
        phate_coords = phate_op.fit_transform(X)
    phate_coordst = torch.tensor(phate_coords)
    phate_D = torch.cdist(phate_coordst, phate_coordst).cpu().detach().numpy()

    trainloader, testloader = train_and_testloader_from_pc(
        X, # <---- Pointcloud
        phate_D, # <---- Distance matrix to match
        batch_size=cfg.data.batch_size,)
    train_sample = next(iter(trainloader))

    activation_dict = {
        'relu': torch.nn.ReLU(),
        'leaky_relu': torch.nn.LeakyReLU(),
        'sigmoid': torch.nn.Sigmoid()
    }

    activation_fn = activation_dict[cfg.model.activation]
    model = AEDist(
        dim=train_sample['x'].shape[1],
        emb_dim=emb_dim,
        layer_widths=cfg.model.layer_widths,
        activation_fn=activation_fn,
        dist_reconstr_weights=cfg.model.dist_reconstr_weights,
        log_dist=cfg.model.log_dist,
        lr=cfg.model.lr,
        dist_recon_topk_coords=cfg.model.dist_recon_topk_coords,
    )
    early_stopping = EarlyStopping(cfg.training.monitor, patience=cfg.training.patience)
    if cfg.logger.use_wandb:
        logger = WandbLogger()
        checkpoint_callback = ModelCheckpoint(
            dirpath=wandb.run.dir,  # Save checkpoints in wandb directory
            save_top_k=3,  # Save the top 3 models
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
        val_dataloaders=testloader,
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

    colors = pd.read_csv(cfg.data.colorpath, index_col=0).iloc[:, 0].values if cfg.data.colorpath is not None else 'b'
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
    _, xh, disparity = procrustes.fit_transform(X, x_hat)
    if cfg.logger.use_wandb:
        wandb.log({'procrustes_disparity_reconstruction': disparity})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.scatter(X[:,0], X[:,1], c=colors, s=1, cmap='Spectral')
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
        plt.savefig(f'{plotdir}/comparison_reconstr.pdf', dpi=300)

    if cfg.logger.use_wandb:
        wandb.log({'Comparison Plot Reconstruction': plt})
        run.finish()

if __name__ == "__main__":
    main()