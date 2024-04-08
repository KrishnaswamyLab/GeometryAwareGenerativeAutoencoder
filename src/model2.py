"""
Train encoder and decoder separately.
By the way refactor and clean up.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from abc import ABC, abstractmethod
from transformations import NonTransform

activation_dict = {
    'relu': torch.nn.ReLU(),
    'leaky_relu': torch.nn.LeakyReLU(),
    'sigmoid': torch.nn.Sigmoid()
}

class MLP(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layer_widths = cfg.get("layer_widths", [[64, 64, 64]])
        assert len(layer_widths) < 2, "layer_widths list must contain at least 2 elements"
        dim = cfg.get("in_dim")
        assert dim is not None, "dim must be specified"
        out_dim = cfg.get("out_dim")
        assert out_dim is not None, "out_dim must be specified"
        activation = cfg.get("activation", "relu")
        assert activation in activation_dict.keys(), f"activation must be one of {list(activation_dict.keys())}"
        batch_norm = cfg.get("batch_norm", False)
        dropout = cfg.get("dropout", 0.0)

        layers = []
        for i, width in enumerate(layer_widths):
            if i == 0:  # First layer, input dimension to first layer width
                layers.append(torch.nn.Linear(dim, width))
            else:  # Subsequent layers, previous layer width to current layer width
                layers.append(torch.nn.Linear(layer_widths[i-1], width))

            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(width))
            activation_func = activation_dict[activation]
            layers.append(activation_func)

            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            
        layers.append(torch.nn.Linear(layer_widths[-1], out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class Encoder(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        cfg.preprocessing.mean = cfg.preprocessing.get('mean', 0.)
        cfg.preprocessing.std = cfg.preprocessing.get('std', 1.)
        cfg.preprocessing.dist_std = cfg.preprocessing.get('dist_std', 1.)
        cfg.loss.dist_mse_decay = cfg.loss.get('dist_mse_decay', 0.)
        self.save_hyperparameters(cfg)
        self.mlp = MLP(cfg.encoder)
        
    def forward(self, x):
        x = (x - self.hparams.preprocessing.mean) / self.hparams.preprocessing.std
        return self.mlp(x)
    
    def loss_function(self, dist_gt, z):
        dist_gt = dist_gt / self.hparams.preprocessing.dist_std
        dist_emb = torch.nn.functional.pdist(z)
        if self.hparams.loss.dist_mse_decay > 0.:
            return ((dist_emb - dist_gt)**2 * torch.exp(-self.hparams.loss.dist_mse_decay * dist_gt)).mean()
        else:
            return torch.nn.functional.mse_loss(dist_emb, dist_gt)

    def step(self, batch, batch_idx):
        x = batch['x']
        d = batch['d']
        loss = self.loss_function(d, self.forward(x))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.training.lr, weight_decay=self.hparams.training.weight_decay)
        return optimizer


class Decoder(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        cfg.preprocessing.mean = cfg.preprocessing.get('mean', 0.)
        cfg.preprocessing.std = cfg.preprocessing.get('std', 1.)
        cfg.preprocessing.dist_std = cfg.preprocessing.get('dist_std', 1.)
        self.save_hyperparameters(cfg)
        self.mlp = MLP(cfg.decoder)
        
    def forward(self, z):
        x = self.mlp(z)
        return x * self.hparams.preprocessing.std + self.hparams.preprocessing.mean
    
    def loss_function(self, z, zhat):
        z = (z - self.hparams.preprocessing.mean) / self.hparams.preprocessing.std
        zhat = (zhat - self.hparams.preprocessing.mean) / self.hparams.preprocessing.std
        return torch.nn.functional.mse_loss(z, zhat)

    def step(self, batch, batch_idx):
        x = batch['x']
        z = batch['z']
        loss = self.loss_function(z, self.forward(x))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.training.lr, weight_decay=self.hparams.training.weight_decay)
        return optimizer
