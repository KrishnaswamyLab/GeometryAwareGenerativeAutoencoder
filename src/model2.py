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
        cfg.encoder.in_dim = cfg.dimensions.get('data')
        cfg.encoder.out_dim = cfg.dimensions.get('latent')
        self.save_hyperparameters(cfg)
        self.mlp = MLP(cfg.encoder)
        
    def normalize(self, x):
        return (x - self.hparams.preprocessing.mean) / self.hparams.preprocessing.std

    def normalize_dist(self, d):
        return d / self.hparams.preprocessing.dist_std

    def forward(self, x, normalize=True): # takes in unnormalized data.
        if normalize:
            x = self.normalize(x)
        return self.mlp(x)
    
    def loss_function(self, dist_gt_norm, z): # assume normalized.
        dist_emb = torch.nn.functional.pdist(z)
        if self.hparams.loss.dist_mse_decay > 0.:
            return ((dist_emb - dist_gt_norm)**2 * torch.exp(-self.hparams.loss.dist_mse_decay * dist_gt_norm)).mean()
        else:
            return torch.nn.functional.mse_loss(dist_emb, dist_gt_norm)

    def step(self, batch, batch_idx, stage):
        x = batch['x']
        d = batch['d']
        z = self.forward(x)
        d_norm = self.normalize_dist(d)
        loss = self.loss_function(d_norm, z)
        self.log(f'{stage}/loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'validation')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'test')
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
        cfg.decoder.in_dim = cfg.dimensions.get('latent')
        cfg.decoder.out_dim = cfg.dimensions.get('data')
        self.save_hyperparameters(cfg)
        self.mlp = MLP(cfg.decoder)
        self.encoder = None
        self.use_encoder = False
    
    def set_encoder(self, encoder):
        self.encoder = encoder
        self.use_encoder = True
        
    def unnormalize(self, x):
        return x * self.hparams.preprocessing.std + self.hparams.preprocessing.mean

    def forward(self, z, unnormalize=True): # outputs unnormalized data
        x = self.mlp(z)
        if unnormalize:
            x = self.unnormalize(x)
        return x

    def loss_function(self, x_norm, xhat_norm): # assume normalized.
        return torch.nn.functional.mse_loss(x_norm, xhat_norm)

    def step(self, batch, batch_idx, stage):
        x = batch['x']
        if self.use_encoder:
            assert self.encoder is not None
            with torch.no_grad():
                z = self.encoder(x) # TODO check if this gives expected behavior.
        else:
            assert 'z' in batch.keys()
            z = batch['z'] # after encoder is trained, do we need to make a new dataset?
        x = self.normalize(x)
        xhat = self.forward(z, unnormalize=False)
        loss = self.loss_function(x, xhat)
        self.log(f'{stage}/loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'validation')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'test')
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.training.lr, weight_decay=self.hparams.training.weight_decay)
        return optimizer

class Autoencoder(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.encoder.in_dim == cfg.decoder.out_dim
        assert cfg.encoder.out_dim == cfg.decoder.in_dim

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def end2end_step(self, batch, batch_idx, stage):
        x = batch['x']
        d = batch['d']
        x_norm = self.encoder.normalize(x)
        zhat = self.encoder(x)
        d_norm = self.encoder.normalize_dist(d)
        xhat_norm = self.decoder(zhat, unnormalize=False)
        loss = self.loss_function(xhat_norm, x_norm, zhat, d_norm, stage)
        return loss

    def loss_function(self, xhat_norm, x_norm, zhat, d_norm, stage):
        """output are the outputs of forward method"""
        # x, x_hat: [B, D]; z: [B, emb_dim]; gt_dist: [B, (B-1)/2]
        loss = 0.0

        if self.hparams.loss.weights.dist > 0.0:
            dl = self.encoder.loss_function(d_norm, zhat)
            self.log(f'{stage}/dist_loss', dl, prog_bar=True, on_epoch=True)
            loss += self.hparams.loss.weights.dist * dl

        if self.hparams.loss.weights.reconstr > 0.0:
            rl = self.decoder.loss_function(x_norm, xhat_norm)
            self.log(f'{stage}/reconstr_loss', rl, prog_bar=True, on_epoch=True)
            loss += self.hparams.loss.weights.reconstr * rl

        if self.hparams.loss.weights.cycle + self.hparams.loss.weights.cycle_dist > 0.0:
            z2 = self.encoder(xhat_norm, normalize=False)
            if self.hparams.loss.weights.cycle > 0.0:
                l2 = torch.nn.functional.mse_loss(zhat, z2)
                self.log(f'{stage}/cycle_loss', l2, prog_bar=True, on_epoch=True)
                loss += self.hparams.loss.weights.cycle * l2
            if self.hparams.loss.weights.cycle_dist > 0.0:
                l3 = self.encoder.loss_function(d_norm, z2)
                self.log(f'{stage}/cycle_dist_loss', l3, prog_bar=True, on_epoch=True)
                loss += self.hparams.loss.weights.cycle_dist * l3
        return loss
    
    def step(self, batch, batch_idx, stage):
        if self.hparams.training.mode == 'end2end':
            loss = self.end2end_step(batch, batch_idx, stage)
        elif self.hparams.training.mode == 'encoder':
            loss = self.encoder.step(batch, batch_idx, f'{stage}_encoder')
        elif self.hparams.training.mode == 'decoder':
            loss = self.decoder.step(batch, batch_idx, f'{stage}_decoder')
        else:
            raise ValueError(f"Invalid training mode: {self.hparams.training.mode}")
        self.log(f'{stage}/loss', loss, prog_bar=True, on_epoch=True)
        return loss
        
    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'validation')
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'test')
        return loss
    
    def configure_optimizers(self):
        if self.training_mode in ['end2end', 'encoder']:
            optimizer_a = torch.optim.Adam(self.encoder.parameters(), lr=self.hparams.training.lr, weight_decay=self.hparams.training.weight_decay)
        if self.training_mode in ['end2end', 'decoder']:
            optimizer_b = torch.optim.Adam(self.decoder.parameters(), lr=self.hparams.training.lr, weight_decay=self.hparams.training.weight_decay)
        
        if self.training_mode == 'end2end':
            return [optimizer_a, optimizer_b], []  # Add schedulers if needed
        elif self.training_mode == 'encoder':
            return optimizer_a
        elif self.training_mode == 'decoder':
            return optimizer_b
