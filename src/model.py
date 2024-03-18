"""
Adapted from https://github.com/KrishnaswamyLab/geosink-te and https://github.com/professorwug/autometric
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from abc import ABC, abstractmethod
from transformations import NonTransform
from heat_kernel import HeatKernelGaussian

class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, layer_widths=[64, 64, 64], activation_fn=torch.nn.ReLU(), dropout=0.0, batch_norm=False):
        super().__init__()
        if out_dim is None:
            out_dim = dim // 2
        if len(layer_widths) < 2:
            raise ValueError("layer_widths list must contain at least 2 elements")

        layers = []
        for i, width in enumerate(layer_widths):
            if i == 0:  # First layer, input dimension to first layer width
                layers.append(torch.nn.Linear(dim, width))
            else:  # Subsequent layers, previous layer width to current layer width
                layers.append(torch.nn.Linear(layer_widths[i-1], width))

            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(width))

            layers.append(activation_fn)

            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))

        layers.append(torch.nn.Linear(layer_widths[-1], out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class BaseAE(pl.LightningModule, ABC):
    def __init__(self, dim, emb_dim, 
                 layer_widths=[64, 64, 64], activation_fn=torch.nn.ReLU(), 
                 eps=1e-10, lr=1e-3, weight_decay=0.0, dropout=0.0, batch_norm=False):
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim
        self.encoder = MLP(dim, emb_dim, 
                           layer_widths=layer_widths, 
                           activation_fn=activation_fn, dropout=dropout, batch_norm=batch_norm)
        self.decoder = MLP(emb_dim, dim, 
                           layer_widths=layer_widths[::-1], 
                           activation_fn=activation_fn, dropout=dropout, batch_norm=batch_norm) # reverse the widths for decoder
        self.eps = eps
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

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
    def loss_function(self, input, output, stage):
        """output are the outputs of forward method"""
        pass

    def step(self, batch, batch_idx, stage):
        x = batch['x']
        d = batch['d']
        input = [x, d]
        loss = self.loss_function(input, self.forward(x), stage)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'train')
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'validation')
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'test')
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

class Decoder(torch.nn.Module):
    def __init__(self, dim, emb_dim, layer_widths=[64, 64, 64], activation_fn=torch.nn.ReLU()):
        super().__init__()
        self.decoder = MLP(emb_dim, dim, 
                           layer_widths=layer_widths, 
                           activation_fn=activation_fn) # reverse the widths for decoder layer_widths[::-1]

    def forward(self, x):
        return self.decoder(x)
    
class AEProb(torch.nn.Module):
    def __init__(self, dim, emb_dim, 
                 layer_widths=[64, 64, 64], activation_fn=torch.nn.ReLU(), 
                 prob_method='tstudent', 
                 dist_reconstr_weights=[1.0, 0.0],
                 eps=1e-8):
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim
        self.prob_method = prob_method
        self.dist_reconstr_weights = dist_reconstr_weights

        self.encoder = MLP(dim, emb_dim, 
                           layer_widths=layer_widths, 
                           activation_fn=activation_fn)
        self.decoder = MLP(emb_dim, dim, 
                           layer_widths=layer_widths[::-1], 
                           activation_fn=activation_fn) # reverse the widths for decoder
        
        
        self.eps = eps

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        '''
            Returns:
                x_hat: [B, D]
                z: [B, emb_dim]
        '''
        z = self.encode(x)
        return [self.decode(z), z]
    
    def decoder_loss(self, x, x_hat):
        '''
            x: [B, D]
            x_hat: [B, D]
        '''
        return torch.nn.functional.mse_loss(x, x_hat)
    
    def encoder_loss(self, gt_matrix, pred_matrix, type='kl'):
        '''
            Inputs:
                gt_matrix: [N, N] transition probability matrix
                pred_matrix: [N, N] transition probability matrix
                type: str, one of 'kl', 'mse'
            Returns:
                loss: scalar
        '''
        loss = 0.0
        if type == 'kl':
            log_pred_mat = torch.log(pred_matrix + self.eps)
            loss = torch.nn.functional.kl_div(log_pred_mat, 
                                              (gt_matrix + self.eps), 
                                              reduction='batchmean',
                                              log_target=False)
        elif type == 'mse':
            loss = torch.nn.functional.mse_loss(gt_matrix, pred_matrix)

        return loss
    
    def compute_prob_matrix(self, z, t: int=1, alpha: float = 1.0, bandwidth: float =1.0, knn: int=5):
        ''' 
            Construct the transition probability of the latent space: each row sum to 1.
            z: [N, emb_dim]
            output: [N, N]
        '''
        probs = None
        if self.prob_method == 'gaussian':
            # symmetric Gaussian kernel
            alpha = alpha
            bandwidth = bandwidth
            dist = torch.cdist(z, z, p=2)
            K = torch.exp(-(dist / bandwidth) ** alpha) # [N, N]
            row_sum = torch.sum(K, dim=1, keepdim=True)
            probs = K / row_sum
            # print('Check symmetry: ', torch.allclose(K, K.T))
            # print('Check probs:', probs.sum(dim=1)[:5], 
            #       torch.allclose(probs.sum(dim=1), torch.ones_like(probs.sum(dim=1))))
        elif self.prob_method == 'adjusted_gaussian':
            # KNN adjutable bandwidth Gaussian kernel
            alpha = alpha
            dist = torch.cdist(z, z, p=2)
            # Find the k-nearest neighbors (including self-loops)
            values, _ = torch.topk(dist, knn, largest=False, dim=-1)
            kth = values[:, -1].unsqueeze(1) # [N, 1]
            K = torch.exp(-(dist / kth) ** alpha)
            K = (K + K.T) / 2.0 # symmetrize
            row_sum = torch.sum(K, dim=1, keepdim=True)
            probs = K / row_sum
            # print('Check symmetry: ', torch.allclose(K, K.T))
            # print('Check probs:', probs.sum(dim=1)[:5], 
            #       torch.allclose(probs.sum(dim=1), torch.ones_like(probs.sum(dim=1))))
        elif self.prob_method == 'tstudent':
            dist = torch.cdist(z, z, p=2) ** 2 # [N, N]
            numerator = (1.0 + dist) ** (-1.0)
            row_sum = torch.sum(numerator, dim=1, keepdim=True)
            probs = numerator / row_sum # [N, N]

        elif self.prob_method == 'heat_kernel':
            #heat_op = HeatKernelGaussian(sigma=1.0, alpha=20, t=t) # FIXME: add these as params
            heat_op = HeatKernelGaussian(sigma=1.0, alpha=1, order=10, t=1)
            probs = heat_op(z)

        elif self.prob_method == 'powered_tstudent':
            dist = torch.cdist(z, z, p=2) ** 2 # [N, N]
            numerator = (1.0 + dist) ** (-1.0)
            row_sum = torch.sum(numerator, dim=1, keepdim=True)
            probs = numerator / row_sum # [N, N]
            probs = torch.linalg.matrix_power(probs, t//2)
            #print('check probs:', probs.shape, probs.sum(dim=1)[:10])
        elif self.prob_method == 'phate':
            raise NotImplementedError('PHATE transition probability not implemented yet')
        else:
            raise ValueError('prob_method must be one of gaussian, tstudent, phate, \
                             heat_kernel, powered_tstudent')

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
        pp=NonTransform(),
        eps=1e-10,
        lr=1e-3,
        weight_decay=0.0,
        dropout=0.0,
        batch_norm=False,
        use_dist_mse_decay=False,
        dist_mse_decay=0.1,
        cycle_weight=0.,
        cycle_dist_weight=0.,
        mean=None,
        std=None,
        dist_std=None,
    ):
        super().__init__(dim, emb_dim, layer_widths=layer_widths, activation_fn=activation_fn, eps=eps, lr=lr, weight_decay=weight_decay, dropout=dropout, batch_norm=batch_norm)
        self.pp = pp
        if dist_recon_topk_coords is None or dist_recon_topk_coords > dim or dist_recon_topk_coords <= 0:
            dist_recon_topk_coords = dim
        self.dist_recon_topk_coords = dist_recon_topk_coords
        self.dist_reg_weight = dist_reconstr_weights[0]
        self.reconstr_weight = dist_reconstr_weights[1]
        self.dist_reconstr_weight = dist_reconstr_weights[2]
        assert self.dist_reg_weight + self.reconstr_weight + self.dist_reconstr_weight > 0.0
        self.use_dist_mse_decay = use_dist_mse_decay
        self.dist_mse_decay = dist_mse_decay
        if self.dist_mse_decay == 0.0:
            self.use_dist_mse_decay = False
        self.cycle_weight = cycle_weight
        self.cycle_dist_weight = cycle_dist_weight
        if mean is None:
            mean = 0
        if std is None:
            std = 1.
        if dist_std is None:
            dist_std = 1.
        # mean = mean.reshape(1, dim)
        # std = std.reshape(1, dim)
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32), persistent=True)
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32), persistent=True)
        self.register_buffer('dist_std', torch.tensor(dist_std, dtype=torch.float32), persistent=True)
        self.save_hyperparameters() 

    def normalize(self, x):
        x = (x - self.mean) / self.std
        return x
    
    def unnormalize(self, x):
        x = x * self.std + self.mean
        return x

    def encode(self, x):
        x_normalized = self.normalize(x)
        z = self.encoder(x_normalized)
        return z

    def decode(self, z):
        x_normalized = self.decoder(z)
        x = self.unnormalize(x_normalized)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_normalized = self.decoder(z)  # used decoder() instead of decode() to match the normalized data.
        return [x_normalized, z]

    def loss_function(self, input, output, stage):
        """output are the outputs of forward method"""
        # x, x_hat: [B, D]; z: [B, emb_dim]; gt_dist: [B, (B-1)/2]
        loss = 0.0
        x_hat_normalized, z = output
        x, dist_gt = input
        x_normalized = self.normalize(x)

        if self.dist_reg_weight > 0.0:
            assert len(input) == 2
            dist_emb = torch.nn.functional.pdist(z) # [B, (B-1)/2] 
            ## pp deprecated! should use 'none'!
            dist_emb = self.pp.transform(dist_emb) # assume the ground truth dist is transformed.
            dist_gt = dist_gt / self.dist_std
            dl = self.dist_loss(dist_emb, dist_gt)
            self.log(f'{stage}/dist_loss', dl, prog_bar=True, on_epoch=True)
            loss += self.dist_reg_weight * dl
            # eps = 1e-10
            eps = 0.
            acc = (1-(torch.abs(dist_gt - dist_emb + eps) / (dist_gt + eps)).mean())
            self.log(f'{stage}/dist_accuracy', acc, prog_bar=True, on_epoch=True)

        if self.dist_reconstr_weight > 0.0:
            """
            DEPRECATED.
            """
            NotImplemented
            # only use top k dimensions for distance, to save computation. 
            # This makes sense only if the input is PCA loadings.
            # TODO compute and transform the original distance before training, to speed up!
            # dist_orig = torch.nn.functional.pdist(x_normalized[:, :self.dist_recon_topk_coords])
            # dist_reconstr = torch.nn.functional.pdist(x_hat_normalized[:, :self.dist_recon_topk_coords])
            # dist_orig = self.pp.transform(dist_orig)
            # dist_reconstr = self.pp.transform(dist_reconstr)
            # drl = self.dist_loss(dist_reconstr, dist_orig)
            # self.log(f'{stage}/dist_reconstr_loss', drl, prog_bar=True, on_epoch=True)
            # loss += self.dist_reconstr_weight * drl
        if self.reconstr_weight > 0.0:
            rl = torch.nn.functional.mse_loss(x_normalized, x_hat_normalized)
            self.log(f'{stage}/reconstr_loss', rl, prog_bar=True, on_epoch=True)
            loss += self.reconstr_weight * rl

        return loss

    def dist_loss(self, dist_emb, dist_gt):
        # dist_emb = self.pp.transform(dist_emb)
        # dist_gt = self.pp.transform(dist_gt) # it is already transformed!
        if self.use_dist_mse_decay:
            return ((dist_emb - dist_gt)**2 * torch.exp(-self.dist_mse_decay * dist_gt)).mean()
        else:
            return torch.nn.functional.mse_loss(dist_emb, dist_gt)

    def step(self, batch, batch_idx, stage):
        x = batch['x']
        d = batch['d']
        input = [x, d]
        output = self.forward(x)
        loss = self.loss_function(input, output, stage)
        if (self.cycle_weight + self.cycle_dist_weight) > 0.0:
            xh, z = output
            z2 = self.encode(xh)
            if self.cycle_weight > 0.0:
                l2 = torch.nn.functional.mse_loss(z, z2)
                self.log(f'{stage}/cycle_loss', l2, prog_bar=True, on_epoch=True)
                loss += self.cycle_weight * l2
            if self.cycle_dist_weight > 0.0:
                dist_emb = torch.nn.functional.pdist(z) # [B, (B-1)/2] 
                dist_emb = self.pp.transform(dist_emb) # assume the ground truth dist is transformed.
                dl = self.dist_loss(dist_emb, d)
                self.log(f'{stage}/dist_loss', dl, prog_bar=True, on_epoch=True)
                loss += self.dist_reg_weight * dl
        return loss

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
