"""
Adapted from https://github.com/KrishnaswamyLab/geosink-te and https://github.com/professorwug/autometric
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from abc import ABC, abstractmethod

class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, layer_widths=[64, 64, 64], activation_fn=torch.nn.ReLU()):
        super().__init__()
        if out_dim is None:
            out_dim = dim // 2
        if len(layer_widths) < 2:
            raise ValueError("layer_widths list must contain at least 2 elements")

        layers = [torch.nn.Linear(dim, layer_widths[0]), activation_fn]

        for i in range(1, len(layer_widths)):
            layers.append(torch.nn.Linear(layer_widths[i-1], layer_widths[i]))
            layers.append(activation_fn)

        layers.append(torch.nn.Linear(layer_widths[-1], out_dim))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class BaseAE(pl.LightningModule, ABC):
    def __init__(self, dim, emb_dim, layer_widths=[64, 64, 64], activation_fn=torch.nn.ReLU(), log_dist=False, eps=1e-10, lr=1e-3):
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim
        self.encoder = MLP(dim, emb_dim, layer_widths=layer_widths, activation_fn=activation_fn)
        self.decoder = MLP(emb_dim, dim, layer_widths=layer_widths[::-1], activation_fn=activation_fn) # reverse the widths for decoder
        self.log_dist = log_dist
        self.eps = eps
        self.lr = lr

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def dist_loss(self, dist_emb, dist_gt):
        if self.log_dist:
            dist_emb = torch.log(dist_emb + self.eps)
            dist_gt = torch.log(dist_gt + self.eps)
        return torch.nn.functional.mse_loss(dist_emb, dist_gt)

    @abstractmethod
    def loss_function(self, input, output):
        """output are the outputs of forward method"""
        pass

    def step(self, batch, batch_idx):
        x = batch['x']
        d = batch['d']
        input = [x, d]
        loss = self.loss_function(input, self.forward(x))
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class AEDist(BaseAE):
    def __init__(
        self,
        dim,
        emb_dim,
        layer_widths=[64, 64, 64],
        activation_fn=torch.nn.ReLU(),
        dist_reg=True,
        dist_reconstr_weights=[1.0, 1.0],
        log_dist=False,
        eps=1e-10,
        lr=1e-3,
    ):
        super().__init__(dim, emb_dim, layer_widths=layer_widths, activation_fn=activation_fn, log_dist=log_dist, eps=eps, lr=lr)
        self.dist_reg = dist_reg
        self.dist_reg_weight = dist_reconstr_weights[0]
        self.reconstr_weight = dist_reconstr_weights[1]

    def forward(self, x):
        z = self.encode(x)
        return [self.decode(z), z]

    def loss_function(self, input, output):
        """output are the outputs of forward method"""
        loss = 0.0
        x_hat, z = output
        if self.dist_reg:
            assert len(input) == 2
            x, dist_gt = input
            dist_emb = torch.nn.functional.pdist(z)
            dl = self.dist_loss(dist_emb, dist_gt)
            self.log('dist_loss', dl, prog_bar=True, on_epoch=True)
            loss += self.dist_reg_weight * dl
        else:
            x = input
        rl = torch.nn.functional.mse_loss(x, x_hat)
        self.log('reconstr_loss', rl, prog_bar=True, on_epoch=True)
        loss += self.reconstr_weight * rl
        return loss

    @torch.no_grad()
    def generate(self, x):
        return self.decode(self.encode(x))
    

class VAEDist(BaseAE):
    def __init__(
        self,
        dim,
        emb_dim,
        layer_widths=[64, 64, 64],
        activation_fn=torch.nn.ReLU(),
        dist_reg=True,
        dist_reconstr_weights=[1.0, 1.0],
        log_dist=False,
        eps=1e-10,
        lr=1e-3,
    ):
        super().__init__(dim, emb_dim, layer_widths, activation_fn, log_dist=log_dist, eps=eps, lr=lr)

        self.dist_reg = dist_reg
        self.dist_reg_weight = dist_reconstr_weights[0]
        self.reconstr_weight = dist_reconstr_weights[1]
        self.fc_mu = torch.nn.Linear(emb_dim, emb_dim)
        self.fc_var = torch.nn.Linear(emb_dim, emb_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode_param(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, log_var = self.encode_param(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), z, mu, log_var

    def loss_function(self, input, output):
        """args are the outputs of forward method"""
        loss = 0.0
        x_hat, z, mu, log_var = output
        if self.dist_reg:
            assert (
                len(input) == 2
            )  # input should be the observations and their pairwise distances.
            x, dist_gt = input
            dist_emb = torch.nn.functional.pdist(z)
            loss += self.dist_reg_weight * self.dist_loss(
                dist_emb, dist_gt
            )
        else:
            x = input
        recon_loss = torch.nn.functional.mse_loss(x, x_hat)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss += self.reconstr_weight * (recon_loss + kl_loss)
        return loss

    @torch.no_grad()
    def sample(self, num_samples, device="cpu"):
        z = torch.randn(num_samples, self.emb_dim).to(device)
        return self.decode(z)

    @torch.no_grad()
    def generate(self, x):
        mu, log_var = self.encode_param(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z)
