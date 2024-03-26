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
    run = None
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

    device_av = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.accelerator is None or cfg.accelerator == 'auto':
        device = device_av
    else:
        device = cfg.accelerator
    
    trainer = pl.Trainer(logger=logger, max_epochs=cfg.max_epochs, accelerator=device,
                         callbacks=[early_stopping,checkpoint_callback],
                         log_every_n_steps=50)  # Adjust as per your setup
    trainer.fit(model, train_loader)
    # gs = model.generate_samples(num_samples=3000)
    
    # Visualize samples with original data 
    # Disabled by default. The inference is too slow for large step size.
    if cfg.visualize:
        # Load best model and generate samples
        model = DiffusionModel.load_from_checkpoint(checkpoint_callback.best_model_path)
        samples = model.generate_samples(num_samples=3000)
        train_data = []
        for batch in train_loader:
            train_data.append(batch[0])
        train_data = torch.cat(train_data, dim=0).cpu().numpy()
        test_data = []
        for batch in test_loader:
            test_data.append(batch[0])
        test_data = torch.cat(test_data, dim=0).cpu().numpy()
        samples = samples.cpu().numpy()
        
        print('train_data', train_data.shape)
        print('test_data', test_data.shape)
        print('samples', samples.shape)

        visualize(samples, train_data, test_data, cfg)


def visualize(samples, train_data, test_data, cfg):
    assert train_data.shape[1] <= 3, "Can only visualize 3D or less"

    import matplotlib.pyplot as plt
    import scprep

    # Plot samples, train, test on the same plot
    print("Visualizing generated samples...")
    plt.rcParams['font.family'] = 'serif'
    n_rows = 1
    n_plots_per_row = 2
    fig = plt.figure(figsize=(n_plots_per_row * 7, n_rows * 6))
    size = 5
    fontsize = 6

    # Plot test + train data
    if train_data.shape[1] == 2:
        ax = fig.add_subplot(n_rows, n_plots_per_row, 1)
        scprep.plot.scatter2d(train_data,
                            legend=True,
                            c='blue',
                            ax=ax,
                            xticks=True,
                            yticks=True,
                            fontsize=fontsize,
                            s=size)
        scprep.plot.scatter2d(test_data,
                            legend=True,
                            c='green',
                            ax=ax,
                            xticks=True,
                            yticks=True,
                            fontsize=fontsize,
                            s=size)
    else:
        ax = fig.add_subplot(n_rows, n_plots_per_row, 1, projection='3d')
        scprep.plot.scatter3d(train_data,
                            legend=True,
                            c='blue',
                            ax=ax,
                            xticks=True,
                            yticks=True,
                            label_prefix='',
                            fontsize=fontsize,
                            s=size)
        scprep.plot.scatter3d(test_data,
                            legend=True,
                            c='green',
                            ax=ax,
                            xticks=True,
                            yticks=True,
                            label_prefix='',
                            fontsize=fontsize,
                            s=size)
    
    ax.set_title('Train+Test')
    

    # Plot samples
    if samples.shape[1] == 2:
        ax = fig.add_subplot(n_rows, n_plots_per_row, 2)
        scprep.plot.scatter2d(samples,
                            legend=False,
                            ax=ax,
                            xticks=True,
                            yticks=True,
                            label_prefix='',
                            fontsize=fontsize,
                            s=size)
    else:
        ax = fig.add_subplot(n_rows, n_plots_per_row, 2, projection='3d')
        scprep.plot.scatter3d(samples,
                            legend=False,
                            ax=ax,
                            xticks=True,
                            yticks=True,
                            label_prefix='',
                            fontsize=fontsize,
                            s=size)
    ax.set_title('Generated Samples')
    
    plt.tight_layout()
    plt.show()
    save_path = cfg.data + "_samples.png"
    plt.savefig(save_path)

    print("Saved Samples to %s" % (save_path))


def load_data(cfg):
    # Load and scale data
    path = cfg.path + cfg.data
    data = np.load(path, allow_pickle=True)
    data_keys = []
    for k, _ in data.items():
        data_keys.append(k)

    all_data = torch.tensor(data['data'], dtype=torch.float32)
    data_size = all_data.size(1)
    if 'is_train' in data_keys:
        train_mask = data['is_train']
    else:
        train_mask = data['train_mask']
    if 'bool' not in train_mask.dtype.name:
        train_mask = train_mask.astype(bool)
    # color_data = torch.tensor(data['colors'], dtype=torch.float32)
    # all_data = torch.concat([all_data, color_data.reshape(-1,1)], dim=1)
    mean_val = torch.mean(all_data, axis=0)
    std_val = torch.std(all_data, axis=0)
    # Standardize data [TODO] need to save the mean and std!
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