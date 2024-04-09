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
# from model import AEDist, VAEDist
from model2 import Encoder, Decoder
from metrics import distance_distortion, mAP
from procrustes import Procrustes

@hydra.main(version_base=None, config_path='../config', config_name='config')
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

    trainloader, valloader, X, phate_coords, colors, dist, mean, std, dist_std = load_data(cfg)
    cfg.preprocessing.mean = mean
    cfg.preprocessing.std = std
    cfg.preprocessing.dist_std = dist_std
    cfg.dimensions.data = X.shape[1]
    model = Autoencoder(cfg)
    early_stoppingEnc = EarlyStopping(cfg.training.monitor, patience=cfg.training.patience)
    if cfg.logger.use_wandb:
        logger = WandbLogger()
        checkpoint_callbackEnc = ModelCheckpoint(
            dirpath=wandb.run.dir,  # Save checkpoints in wandb directory
            save_top_k=1,  # Save the top 1 model
            monitor='validation/loss',  # Model selection based on validation loss
            mode='min'  # Minimize validation loss
        )
    else:
        logger = TensorBoardLogger(save_dir=os.path.join(cfg.path.root, cfg.path.log))
        checkpoint_callbackEnc = ModelCheckpoint(
            dirpath=cfg.path.root,  # Save checkpoints in wandb directory
            filename=cfg.path.model,
            save_top_k=1,
            monitor='validation/loss',  # Model selection based on validation loss
            mode='min'  # Minimize validation loss
        )
    trainer = Trainer(
        logger=logger,
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        callbacks=[early_stoppingEnc,checkpoint_callbackEnc],
        log_every_n_steps=cfg.training.log_every_n_steps,
    )
 
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=valloader,
    )

    if cfg.logger.use_wandb:
        run.finish()

def load_data(cfg):
    data_path = os.path.join(cfg.data.root, cfg.data.name + cfg.data.filetype)
    data = np.load(data_path, allow_pickle=True)
    X = data['data']
    phate_coords = data['phate']
    colors = data['colors']
    dist = data['dist']
    dist_std = np.std(dist.flatten())
    train_mask = data['is_train'].astype(bool) # !!! Fixed bug: when mask is not boolean it is problematic!
    X = X[train_mask,:]
    phate_coords = phate_coords[train_mask,:]
    colors = colors[train_mask]
    dist = dist[train_mask,:][:,train_mask]
    trainloader, valloader, mean, std = train_valid_loader_from_pc(
        X, # <---- Pointcloud
        dist, # <---- Distance matrix to match
        batch_size=cfg.training.batch_size,
        train_valid_split=cfg.training.train_valid_split,
        shuffle=cfg.training.shuffle,
        seed=cfg.training.seed, return_mean_std=True, componentwise_std=cfg.model.componentwise_std)
    return trainloader, valloader, X, phate_coords, colors, dist, mean, std, dist_std
