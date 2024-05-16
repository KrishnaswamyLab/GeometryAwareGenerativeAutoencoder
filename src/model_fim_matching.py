import torch
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad, Variable
from torch.nn.utils import spectral_norm

from model2 import Encoder

"""
From https://github.com/xingzhis/dmae/blob/6734618fc1f72b1f0fcb4e328d0a7101f1f142ba/src/model.py
"""
def compute_prob_matrix(z, t: int=1, alpha: float = 1.0, bandwidth: float =1.0, knn: int=5, prob_method: str='gaussian'):
    ''' 
        Construct the transition probability of the latent space: each row sum to 1.
        z: [N, emb_dim]
        output: [N, N]
    '''
    probs = None
    if prob_method == 'gaussian':
        # Gaussian kernel
        alpha = alpha
        bandwidth = bandwidth
        dist = torch.cdist(z, z, p=2)
        K = torch.exp(-(dist / bandwidth) ** alpha) # [N, N]
        row_sum = torch.sum(K, dim=1, keepdim=True)
        probs = K / row_sum
        # print('Check symmetry: ', torch.allclose(K, K.T))
        # print('Check probs:', probs.sum(dim=1)[:5], 
        #       torch.allclose(probs.sum(dim=1), torch.ones_like(probs.sum(dim=1))))
    elif prob_method == 'sym_gaussian':
        # Gaussian kernel
        alpha = alpha
        bandwidth = bandwidth
        dist = torch.cdist(z, z, p=2)
        K = torch.exp(-(dist / bandwidth) ** alpha)
        K = (K + K.T) / 2.0 # symmetrize
        row_sum = torch.sum(K, dim=1, keepdim=True)
        probs = K / row_sum

    elif prob_method == 'adjusted_gaussian':
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
    elif prob_method == 'tstudent':
        dist = torch.cdist(z, z, p=2) ** 2 # [N, N]
        numerator = (1.0 + dist) ** (-1.0)
        row_sum = torch.sum(numerator, dim=1, keepdim=True)
        probs = numerator / row_sum # [N, N]

    elif prob_method == 'heat_kernel':
        #heat_op = HeatKernelGaussian(sigma=1.0, alpha=20, t=t) # FIXME: add these as params
        heat_op = HeatKernelGaussian(sigma=1.0, alpha=1, order=10, t=1)
        probs = heat_op(z)

    elif prob_method == 'powered_tstudent':
        dist = torch.cdist(z, z, p=2) ** 2 # [N, N]
        numerator = (1.0 + dist) ** (-1.0)
        row_sum = torch.sum(numerator, dim=1, keepdim=True)
        probs = numerator / row_sum # [N, N]
        probs = torch.linalg.matrix_power(probs, t//2)
        #print('check probs:', probs.shape, probs.sum(dim=1)[:10])
    elif prob_method == 'phate':
        raise NotImplementedError('PHATE transition probability not implemented yet')
    else:
        raise ValueError('prob_method must be one of gaussian, tstudent, phate, \
                            heat_kernel, powered_tstudent')

    return probs

class FIMMEncoder(Encoder):
    def __init__(self, cfg, preprocessor):
        super().__init__(cfg, preprocessor)

    def loss_function(self, dist_gt_norm, z, mask): # don't use the mask any more.
        q = compute_prob_matrix(z, t=self.hparams.cfg.fimm.t, alpha=self.hparams.cfg.fimm.alpha, bandwidth=self.hparams.cfg.fimm.bandwidth, knn=self.hparams.cfg.fimm.knn, prob_method=self.hparams.cfg.fimm.prob_method)
        logq = torch.log(q + 1e-8)
        dist_emb = torch.nn.functional.pdist(logq)

        if self.hparams.cfg.loss.dist_mse_decay > 0.:
            return (torch.square(dist_emb - dist_gt_norm) * torch.exp(-self.hparams.cfg.loss.dist_mse_decay * dist_gt_norm)).mean()
        else:
            return torch.nn.functional.mse_loss(dist_emb, dist_gt_norm)

class JSDMEncoder(Encoder):
    def __init__(self, cfg, preprocessor):
        super().__init__(cfg, preprocessor)
        self.compute_prob_matrix = lambda z: compute_prob_matrix(z, t=self.hparams.cfg.fimm.t, alpha=self.hparams.cfg.fimm.alpha, bandwidth=self.hparams.cfg.fimm.bandwidth, knn=self.hparams.cfg.fimm.knn, prob_method=self.hparams.cfg.fimm.prob_method)

    def loss_function(self, p, z, mask):
        """
        the step(self, batch, batch_idx, stage) will take d = batch['d'],
        but we put the p in the place of 'd' for compatibility (because they have the same shape),
        so the "dist" parameter is actually p
        """
        q = self.compute_prob_matrix(z)
        m = 0.5 * (p + q)
        logq = torch.log(q + 1e-7)
        logp = torch.log(p + 1e-7)
        logm = torch.log(m + 1e-7)
        kl1 = p * (logp - logm)
        kl2 = q * (logq - logm)
        jsd = 0.5 * (kl1 + kl2)
        return jsd