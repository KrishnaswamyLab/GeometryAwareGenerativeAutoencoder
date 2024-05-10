import os
import wandb
import hydra
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from data import train_valid_loader_from_pc
from model2 import Autoencoder, Preprocessor, Discriminator, NoisePredictor, WDiscriminator

@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # This prints the entire configuration to the console
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

    trainloader, valloader, X, phate_coords, colors, preprocessor = load_data(cfg)
    cfg.dimensions.data = X.shape[1]
    if cfg.training.mode == 'discriminator':
        model = Discriminator(cfg, preprocessor)
    elif cfg.training.mode == 'noise_predictor':
        model = NoisePredictor(cfg, preprocessor)
    elif cfg.training.mode == 'wdiscriminator':
        model = WDiscriminator(cfg, preprocessor)
    else:
        model = Autoencoder(cfg, preprocessor)

    if cfg.logger.use_wandb:
        logger = WandbLogger()
        checkpoint_callback = ModelCheckpoint(
            dirpath=wandb.run.dir,  # Save checkpoints in wandb directory
            save_top_k=1,  # Save the top 1 model
            monitor=cfg.training.monitor,  # Model selection based on validation loss
            mode='min'  # Minimize validation loss
        )
    else:
        logger = TensorBoardLogger(save_dir=os.path.join(cfg.path.root, cfg.path.log))
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.path.root,  # Save checkpoints in wandb directory
            filename=cfg.path.model,
            save_top_k=1,
            monitor=cfg.training.monitor,  # Model selection based on validation loss
            mode='min'  # Minimize validation loss
        )

    if cfg.training.mode in ['discriminator', 'wdiscriminator', 'noise_predictor']:
        train_model(cfg, model, trainloader, valloader, logger, checkpoint_callback)
    elif cfg.training.mode == 'separate':
        cfg.training.mode = 'encoder'
        train_model(cfg, model.encoder, trainloader, valloader, logger, checkpoint_callback)
        model.link_encoder()
        cfg.training.mode = 'decoder'
        train_model(cfg, model.decoder, trainloader, valloader, logger, checkpoint_callback)
    elif cfg.training.mode in ['end2end', 'negative', 'radius']: # encoder-only, decoder-only, or end-to-end
        train_model(cfg, model, trainloader, valloader, logger, checkpoint_callback)
    elif cfg.training.mode == 'encoder':
        train_model(cfg, model.encoder, trainloader, valloader, logger, checkpoint_callback)
    elif cfg.training.mode == 'decoder':
        train_model(cfg, model.decoder, trainloader, valloader, logger, checkpoint_callback)
    else:
        raise ValueError('Invalid training mode')

    if cfg.logger.use_wandb:
        run.finish()

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

def load_data(cfg):
    data_path = os.path.join(cfg.data.root, cfg.data.name + cfg.data.filetype)
    data = np.load(data_path, allow_pickle=True)
    X = data['data'].astype(np.float32)
    # phate_coords = data['phate'].astype(np.float32)
    dist = data['dist'].astype(np.float32)
    dist_std = np.std(dist.flatten())
    train_mask = data['is_train'].astype(bool) # !!! Fixed bug: when mask is not boolean it is problematic!
    X = X[train_mask,:]
    # phate_coords = phate_coords[train_mask,:]
    phate_coords = None
    if 'colors' in data.files:
        colors = data['colors']
        colors = colors[train_mask]
    else:
        colors = None
    dist = dist[train_mask,:][:,train_mask]
    mask_x = data.get('mask_x', None)
    mask_d = data.get('mask_d', None)
    if mask_x is not None:
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
        batch_size=cfg.training.batch_size,
        train_valid_split=cfg.training.train_valid_split,
        shuffle=cfg.training.shuffle,
        seed=cfg.training.seed, return_mean_std=True, componentwise_std=False)
    preprocessor = Preprocessor(mean=mean, std=std, dist_std=dist_std)
    return trainloader, valloader, X, phate_coords, colors, preprocessor

if __name__ == '__main__':
    main()