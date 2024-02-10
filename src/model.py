"""
Adapted from https://github.com/KrishnaswamyLab/geosink-te and https://github.com/professorwug/autometric
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from abc import ABC, abstractmethod
from transformations import NonTransform

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
    def __init__(self, dim, emb_dim, 
                 layer_widths=[64, 64, 64], activation_fn=torch.nn.ReLU(), 
                 eps=1e-10, lr=1e-3):
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim
        self.encoder = MLP(dim, emb_dim, 
                           layer_widths=layer_widths, 
                           activation_fn=activation_fn)
        self.decoder = MLP(emb_dim, dim, 
                           layer_widths=layer_widths[::-1], 
                           activation_fn=activation_fn) # reverse the widths for decoder
        self.eps = eps
        self.lr = lr

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def dist_loss(self, dist_emb, dist_gt):
        pass

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


class ProbabilityEncoder(torch.nn.Module):
    '''
        Encoder that matches probability distribution of the input data,
        and the latent space.
    '''
    def __init__(self, dim, emb_dim, 
                 layer_widths=[64, 32, 16], activation_fn=torch.nn.ReLU(),
                 prob_method='Guassian'):
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim

        self.encoder = MLP(dim, emb_dim, 
                           layer_widths=layer_widths, 
                           activation_fn=activation_fn)
        
        self.prob_method = prob_method

    def forward(self, x):
        z = self.encoder(x)
        probs = self._transition_prob(z)

        return probs

    def _transition_prob(self, z):
        ''' 
            Construct the transition probability of the latent space: each row sum to 1.
            z: [N, emb_dim]
            output: [N, N]
        '''
        probs = None

        if self.prob_method == 'gaussian':
            raise NotImplementedError('Gaussian transition probability not implemented yet')
        elif self.prob_method == 'tstudent':
            dist = torch.nn.functional.cdist(z, z, p=2) ** 2 # [N, N]
            numerator = (1.0 + dist) ** (-1.0)
            row_sum = torch.sum(numerator, dim=1, keepdim=True)
            probs = numerator / row_sum # [N, N]
        elif self.prob_method == 'phate':
            raise NotImplementedError('PHATE transition probability not implemented yet')
        else:
            raise ValueError('prob_method must be one of gaussian, tstudent, phate')

        return probs
        

class AEDist(BaseAE):
    def __init__(
        self,
        dim,
        emb_dim,
        layer_widths=[64, 64, 64],
        activation_fn=torch.nn.ReLU(),
        dist_reconstr_weights=[0.9, 0.1, 0.0],
        dist_recon_topk_coords=None,
        preprocessor=NonTransform(),
        eps=1e-10,
        lr=1e-3,
        use_dist_mse_decay=False,
        dist_mse_decay=0.1,
    ):
        super().__init__(dim, emb_dim, 
                         layer_widths=layer_widths, activation_fn=activation_fn, 
                         eps=eps, lr=lr)
        self.preprocessor = preprocessor
        
        self.dist_weight = dist_reconstr_weights[0]
        self.reconstr_weight = dist_reconstr_weights[1]
        assert self.dist_weight + self.reconstr_weight > 0.0
        
        self.use_dist_mse_decay = use_dist_mse_decay
        self.dist_mse_decay = dist_mse_decay

    def forward(self, x):
        z = self.encode(x)
        return [self.decode(z), z]

    def loss_function(self, input, output):
        """output are the outputs of forward method"""
        # x, x_hat: [B, D]; z: [B, emb_dim]; gt_dist: [B, (B-1)/2]
        loss = 0.0
        x_hat, z = output
        x, dist_gt = input

        if self.dist_weight > 0.0:
            dist_emb = torch.nn.functional.pdist(z) # [B, (B-1)/2] 
            if self.preprocessor is not None:
                dist_emb = self.preprocessor.transform(dist_emb)

            dl = self.dist_loss(dist_emb, dist_gt)          
            self.log('dist_loss', dl, prog_bar=True, on_epoch=True)
            loss += self.dist_reg_weight * dl
        if self.reconstr_weight > 0.0:
            rl = torch.nn.functional.mse_loss(x, x_hat)
            self.log('reconstr_loss', rl, prog_bar=True, on_epoch=True)
            loss += self.reconstr_weight * rl

        return loss

    def dist_loss(self, dist_emb, dist_gt):
        if self.use_dist_mse_decay:
            return ((dist_emb - dist_gt)**2 * torch.exp(-self.dist_mse_decay * dist_gt)).mean()
        else:
            return torch.nn.functional.mse_loss(dist_emb, dist_gt)

    @torch.no_grad()
    def generate(self, x):
        return self.decode(self.encode(x))
    

class VAEDist(AEDist):
    # TODO add dist_mse_decay to VAE?
    def __init__(        
        self,
        dim,
        emb_dim,
        layer_widths=[64, 64, 64],
        activation_fn=torch.nn.ReLU(),
        dist_reconstr_weights=[0.1, 0.6, 0.3],
        kl_weight=1.0,
        dist_recon_topk_coords=None,
        pp=NonTransform(),
        eps=1e-10,
        lr=1e-3,
    ):
        super().__init__(dim, emb_dim, layer_widths=layer_widths, activation_fn=activation_fn, dist_reconstr_weights=dist_reconstr_weights, dist_recon_topk_coords=dist_recon_topk_coords, pp=pp, eps=eps, lr=lr)
        self.fc_mu = torch.nn.Linear(emb_dim, emb_dim)
        self.fc_var = torch.nn.Linear(emb_dim, emb_dim) 
        self.kl_weight = kl_weight  

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
    
    def forward_proba(self, x):
        mu, log_var = self.encode_param(x)
        z = self.reparameterize(mu, log_var)
        return [self.decoder(z), z, mu, log_var]

    def forward(self, x):
        xhat, z, mu, logvar = self.forward_proba(x)
        return [xhat, z]

    def step(self, batch, batch_idx):
        x = batch['x']
        d = batch['d']
        input = [x, d]
        loss = self.loss_function(input, self.forward_proba(x))
        return loss

    def loss_function(self, input, output):
        """args are the outputs of forward method"""
        x_hat, z, mu, log_var = output
        loss = 0.0
        loss += super().loss_function(input, [x_hat, z])
        # if self.dist_reg:
        #     assert (
        #         len(input) == 2
        #     )  # input should be the observations and their pairwise distances.
        #     x, dist_gt = input
        #     dist_emb = torch.nn.functional.pdist(z)
        #     loss += self.dist_reg_weight * self.dist_loss(
        #         dist_emb, dist_gt
        #     )
        # else:
        #     x = input
        # recon_loss = torch.nn.functional.mse_loss(x, x_hat)
        if self.kl_weight > 0.0:
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            self.log('kl_loss', kl_loss, prog_bar=True, on_epoch=True)
            loss += self.kl_weight * kl_loss
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


"""
[DEPRECATED for now] need to add parameters to fit in the current config. should made a subclass of AEDist.
"""
# class VAEDist(BaseAE):
#     def __init__(
#         self,
#         dim,
#         emb_dim,
#         layer_widths=[64, 64, 64],
#         activation_fn=torch.nn.ReLU(),
#         dist_reg=True,
#         dist_reconstr_weights=[1.0, 1.0],
#         log_dist=False,
#         eps=1e-10,
#         lr=1e-3,
#     ):
#         super().__init__(dim, emb_dim, layer_widths, activation_fn, log_dist=log_dist, eps=eps, lr=lr)

#         self.dist_reg = dist_reg
#         self.dist_reg_weight = dist_reconstr_weights[0]
#         self.reconstr_weight = dist_reconstr_weights[1]
#         self.fc_mu = torch.nn.Linear(emb_dim, emb_dim)
#         self.fc_var = torch.nn.Linear(emb_dim, emb_dim)

#     def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         """
#         Reparameterization trick to sample from N(mu, var) from
#         N(0,1).
#         :param mu: (Tensor) Mean of the latent Gaussian [B x D]
#         :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
#         :return: (Tensor) [B x D]
#         """
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu

#     def encode_param(self, x):
#         result = self.encoder(x)
#         result = torch.flatten(result, start_dim=1)

#         mu = self.fc_mu(result)
#         log_var = self.fc_var(result)

#         return [mu, log_var]

#     def decode(self, x):
#         return self.decoder(x)

#     def forward(self, x):
#         mu, log_var = self.encode_param(x)
#         z = self.reparameterize(mu, log_var)
#         return self.decoder(z), z, mu, log_var

#     def loss_function(self, input, output):
#         """args are the outputs of forward method"""
#         loss = 0.0
#         x_hat, z, mu, log_var = output
#         if self.dist_reg:
#             assert (
#                 len(input) == 2
#             )  # input should be the observations and their pairwise distances.
#             x, dist_gt = input
#             dist_emb = torch.nn.functional.pdist(z)
#             loss += self.dist_reg_weight * self.dist_loss(
#                 dist_emb, dist_gt
#             )
#         else:
#             x = input
#         recon_loss = torch.nn.functional.mse_loss(x, x_hat)
#         kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#         loss += self.reconstr_weight * (recon_loss + kl_loss)
#         return loss

#     @torch.no_grad()
#     def sample(self, num_samples, device="cpu"):
#         z = torch.randn(num_samples, self.emb_dim).to(device)
#         return self.decode(z)

#     @torch.no_grad()
#     def generate(self, x):
#         mu, log_var = self.encode_param(x)
#         z = self.reparameterize(mu, log_var)
#         return self.decode(z)
