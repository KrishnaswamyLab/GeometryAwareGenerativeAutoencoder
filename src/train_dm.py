"""
Train the diffusion model
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import torch.optim as optim
import pytorch_lightning as pl
import os
from diffusion import DiffusionModel

@hydra.main(version_base=None, config_path='../dm_conf', config_name='config')
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

    pl.seed_everything(cfg.seed)

    train_loader, test_loader, data_size = load_data(cfg)
    # Model and optimizer
    model = DiffusionModel(
        data_size=data_size, 
        time_embedding_size=cfg.time_embedding_size,
        layer_widths=cfg.layer_widths,
        dropout=cfg.dropout,
        batch_norm=cfg.batch_norm,
        num_steps=cfg.num_steps,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    early_stopping = EarlyStopping(cfg.monitor, patience=cfg.patience)
    if cfg.logger.use_wandb:
        logger = WandbLogger()
        checkpoint_callback = ModelCheckpoint(
            dirpath=wandb.run.dir,  # Save checkpoints in wandb directory
            save_top_k=1,  # Save the top 1 model
            monitor='train_loss',  # Model selection based on validation loss
            mode='min'  # Minimize validation loss
        )
    else:
        logger = TensorBoardLogger(save_dir=os.path.join(cfg.path.root, cfg.path.log))
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.path.root,  # Save checkpoints in wandb directory
            filename=cfg.path.model,
            save_top_k=1,
            monitor='train_loss',  # Model selection based on validation loss
            mode='min'  # Minimize validation loss
        )
    trainer = pl.Trainer(logger=logger, max_epochs=cfg.max_epochs, gpus=1, 
                         callbacks=[early_stopping,checkpoint_callback],
                         log_every_n_steps=50)  # Adjust as per your setup
    trainer.fit(model, train_loader)
    # gs = model.generate_samples(num_samples=3000)

def load_data(cfg):
    # Load and scale data
    path = cfg.path + cfg.data
    data = np.load(path, allow_pickle=True)
    all_data = torch.tensor(data['data'], dtype=torch.float32)
    data_size = all_data.size(1)
    train_mask = data['train_mask']
    # color_data = torch.tensor(data['colors'], dtype=torch.float32)
    # all_data = torch.concat([all_data, color_data.reshape(-1,1)], dim=1)
    mean_val = torch.mean(all_data, axis=0)
    std_val = torch.std(all_data, axis=0)
    # Standardize data
    scaled_data = (all_data - mean_val) / std_val

    # Split into training and validation sets
    train_data, test_data = scaled_data[train_mask,:], scaled_data[~train_mask,:]

    # Convert to tensor datasets
    train_dataset = TensorDataset(train_data)
    test_dataset = TensorDataset(test_data)

    # Create DataLoaders
    batch_size = cfg.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, data_size

if __name__ == '__main__':
    main()