import torch
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad, Variable
from torch.nn.utils import spectral_norm

from model2 import Encoder

class SphereEncoder(Encoder):
    def __init__(self, cfg, preprocessor):
        super().__init__(cfg, preprocessor)
        self.r = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x, normalize=True):
        z = super().forward(x, normalize)
        z = (z + 1e-8) / (torch.square(z).sum(axis=1) + 1e-8).reshape(-1,1)
        return z

    def loss_function(self, dist_gt_norm, z, mask): # don't use the mask any more.
        # dist_emb = torch.nn.functional.pdist(z)
        dist_emb = z @ z.T
        dist_emb = torch.clamp(dist_emb, -1, 1) # prevent numerical issues.
        dist_emb = self.r * torch.acos(dist_emb)
        dist_emb = dist_emb[np.triu_indices(dist_emb.size(0), k=1)]

        if self.hparams.cfg.loss.dist_mse_decay > 0.:
            return (torch.square(dist_emb - dist_gt_norm) * torch.exp(-self.hparams.cfg.loss.dist_mse_decay * dist_gt_norm)).mean()
        else:
            return torch.nn.functional.mse_loss(dist_emb, dist_gt_norm)

class HyperbolicPoincareEncoder(Encoder):
    NotImplemented
    
def minkowski_inner_product_matrix(z):
    t = z[:, 0:1]
    x = z[:, 1:]
    tt = torch.matmul(t, t.t())
    xx = torch.matmul(x, x.t())
    minkowski_matrix = tt - xx
    return minkowski_matrix

def safe_acosh(x):
    # Clamp values to be >= 1
    x_clamped = torch.clamp(x, min=1)
    # Compute acosh, adding a small epsilon to the clamped values
    return torch.acosh(x_clamped + 1e-8)

class HyperbolicLorenzEncoder(Encoder):
    def __init__(self, cfg, preprocessor):
        super().__init__(cfg, preprocessor)
        self.r = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x, normalize=True):
        z = super().forward(x, normalize)
        z0 = torch.sqrt(1 + torch.sum(z ** 2, dim=-1, keepdim=True) + 1e-8)
        z_hyperboloid = torch.cat([z0, z], dim=-1)
        return z_hyperboloid
    
    def loss_function(self, dist_gt_norm, z, mask): # don't use the mask any more.
        dist_emb = minkowski_inner_product_matrix(z)
        dist_emb = self.r * safe_acosh(dist_emb)
        dist_emb = dist_emb[np.triu_indices(dist_emb.size(0), k=1)]

        if self.hparams.cfg.loss.dist_mse_decay > 0.:
            return (torch.square(dist_emb - dist_gt_norm) * torch.exp(-self.hparams.cfg.loss.dist_mse_decay * dist_gt_norm)).mean()
        else:
            return torch.nn.functional.mse_loss(dist_emb, dist_gt_norm)
