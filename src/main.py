import sys
from data import train_and_testloader_from_pc
from model import AEDist
import numpy as np
import torch
import phate
from heatgeo.embedding import HeatGeo
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from scipy.spatial import procrustes
import scanpy as sc
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import hydra
import os
from omegaconf import DictConfig, OmegaConf

def create_nested_config(flat_config):
    nested_config = {}
    for key, value in flat_config.items():
        parts = key.split('.')
        current_level = nested_config

        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]

        current_level[parts[-1]] = value
    
    return nested_config

@hydra.main(version_base=None, config_path='../conf', config_name='config')
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(
        entity=cfg.logger.entity,
        project=cfg.logger.project,
        tags=cfg.logger.tags,
        reinit=True,
        settings=wandb.Settings(start_method="thread"),
    )

    nested_wandb_config = create_nested_config(dict(wandb.config))
    print('nested______:')
    print(nested_wandb_config)
    print('cfg______:')
    print(cfg)
    print('wandb.config______:')
    print(wandb.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.create(nested_wandb_config))
    print('merged______:')
    print(cfg)
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)

    if cfg.data.file_type == 'h5ad':
        adata = sc.read_h5ad(cfg.data.datapath)
        X = adata.X[:,:].toarray()
    elif cfg.data.file_type == 'npy':
        X = np.load(cfg.data.datapath)
    else:
        raise ValueError('Unknown file type')
    phate_op = phate.PHATE(
        random_state=cfg.phate.random_state,
        n_components=cfg.model.emb_dim,
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

    if cfg.model.activation == 'relu':
        activation_fn = torch.nn.ReLU()
    elif cfg.model.activation == 'leaky_relu':
        activation_fn = torch.nn.LeakyReLU()
    elif cfg.model.activation == 'sigmoid':
        activation_fn = torch.nn.Sigmoid()
    else:
        raise ValueError('Unknown activation function')

    model = AEDist(
        dim=train_sample['x'].shape[1],
        emb_dim=cfg.model.emb_dim,
        log_dist=cfg.model.log_dist,
        w=cfg.model.w,
        lr=cfg.model.lr,
        activation_fn=activation_fn,
    )
    early_stopping = EarlyStopping(cfg.training.monitor, patience=cfg.training.patience)
    wandb_logger = WandbLogger()
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb.run.dir,  # Save checkpoints in wandb directory
        save_top_k=3,  # Save the top 3 models
        monitor='val_loss',  # Model selection based on validation loss
        mode='min'  # Minimize validation loss
    )
    trainer = Trainer(
        logger=wandb_logger,
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
    emb_z = model(X_tensor)[1].cpu().detach().numpy()

    pc_s, z, disparity = procrustes(phate_coords, emb_z)

    wandb.log({'procrustes_disparity': disparity})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.scatter(phate_coords[:,0], phate_coords[:,1])
    ax1.set_title('PHATE')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.scatter(z[:,0], z[:,1])
    ax2.set_title('Latent Space')
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.suptitle('Comparison of PHATE and Latent Space')

    wandb.log({'Comparison Plot': plt})

    run.finish()

if __name__ == "__main__":
    main()