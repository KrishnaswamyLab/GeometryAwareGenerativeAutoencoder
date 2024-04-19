import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad
adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
import torch.optim as optim
from torch.autograd.functional import jacobian
import pytorch_lightning as pl
import warnings

def compute_jacobian_function(f, x, create_graph=True, retain_graph=True):
    """
    Compute the Jacobian of the f wrt a batch of points in the latent space using an efficient broadcasting approach.
    Args:
        f: The function to compute the Jacobian of. f: (B, D) -> (B, n).
        x: (B, D) A batch of points in the dim D.
    Returns:
        jacobian: (B, n, D) The Jacobian of f wrt x.
    """
    # z_batch = z_batch.clone().detach().requires_grad_(True)
    x = x.clone()
    x.requires_grad_(True)
    # model.no_grad()
    output = f(x)
    batch_size, output_dim, input_dim = *output.shape, x.shape[-1]

    # Use autograd's grad function to get gradients for each output dimension
    jacobian = torch.zeros(batch_size, output_dim, input_dim).to(x.device)
    for i in range(output_dim):
        grad_outputs = torch.zeros(batch_size, output_dim).to(x.device)
        grad_outputs[:, i] = 1.0
        gradients = grad(outputs=output, inputs=x, grad_outputs=grad_outputs, create_graph=create_graph, retain_graph=retain_graph, only_inputs=True)[0]
        jacobian[:, i, :] = gradients
    return jacobian

def pullback_metric(x, fcn, create_graph=True, retain_graph=True, pseudoinverse=False):
    jac = compute_jacobian_function(fcn, x, create_graph, retain_graph)
    if pseudoinverse:
        jac = torch.linalg.pinv(jac)
    metric = torch.einsum('Nki,Nkj->Nij', jac, jac)
    return metric

class ODEFunc(nn.Module):
    """
    For simplicity we are just using 2 layers but it might worth to substitute with the MLP class
    although the torchdiffeq suggusted using tanh activation which we might want to tune.
    """
    def __init__(self, in_dim, hidden_dim):
        super(ODEFunc, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.in_dim),
        )
    def forward(self, t, x):
        return self.net(x)

# class MLP(torch.nn.Module):
#     def __init__(self, dim, out_dim=None, layer_widths=[64, 64, 64], activation_fn=torch.nn.ReLU(), dropout=0.0, batch_norm=False):
#         super().__init__()
#         if out_dim is None:
#             out_dim = dim // 2
#         if len(layer_widths) < 2:
#             raise ValueError("layer_widths list must contain at least 2 elements")

#         layers = []
#         for i, width in enumerate(layer_widths):
#             if i == 0:  # First layer, input dimension to first layer width
#                 layers.append(torch.nn.Linear(dim, width))
#             else:  # Subsequent layers, previous layer width to current layer width
#                 layers.append(torch.nn.Linear(layer_widths[i-1], width))

#             if batch_norm:
#                 layers.append(torch.nn.BatchNorm1d(width))

#             layers.append(activation_fn)

#             if dropout > 0:
#                 layers.append(torch.nn.Dropout(dropout))

#         layers.append(torch.nn.Linear(layer_widths[-1], out_dim))
#         self.net = torch.nn.Sequential(*layers)

#     def forward(self, x):
#         return self.net(x)


class GeodesicODE(pl.LightningModule):
    def __init__(self, 
        fcn, # encoder/decoder
        in_dim=2, 
        hidden_dim=64, 
        n_tsteps=1000, # num of t steps for length evaluation
        lam=10, # regularization for end point
        # layer_widths=[64, 64, 64], 
        # activation_fn=torch.nn.ReLU(), 
        lr=1e-3, 
        weight_decay=0.0, 
        # dropout=0.0, 
        # batch_norm=False,
        beta=0.,
        n_pow=4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.odefunc = ODEFunc(in_dim, hidden_dim)
        self.pretraining = False
        self.t = torch.linspace(0, 1, self.hparams.n_tsteps)


    def length_loss(self, t, x):
        x_flat = x.view(-1, x.shape[2])
        metric_flat = pullback_metric(x_flat, self.hparams.fcn, create_graph=True, retain_graph=True)
        xdot = self.odefunc(t, x)
        xdot_flat = xdot.view(-1, xdot.shape[2])
        l_flat = torch.sqrt(torch.einsum('Ni,Nij,Nj->N', xdot_flat, metric_flat, xdot_flat))
        return l_flat.mean()# * (t[-1] - t[0]) # numerical integration, we set t in [0,1].
    
    def forward(self, x0):
        t = self.t
        x_t = odeint(self.odefunc, x0, t)
        return x_t

    def step(self, batch, batch_idx):
        t = self.t
        x0, x1 = batch
        x_t = self.forward(x0)
        mse_loss = F.mse_loss(x_t[-1], x1)
        if self.pretraining:
            return mse_loss
        mpowerede_loss = 0.
        if self.hparams.beta > 0.:
            mpowerede_loss = (torch.pow(x_t[-1] - x1, self.hparams.n_pow)).mean() * self.hparams.beta
        len_loss = self.length_loss(t, x_t)
        loss = len_loss + self.hparams.lam * mse_loss + mpowerede_loss
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
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

class GeodesicODEDensity(GeodesicODE):
    def __init__(self, 
        fcn, # encoder/decoder
        in_dim=2, 
        hidden_dim=64, 
        n_tsteps=1000, # num of t steps for length evaluation
        lam=10, # regularization for end point
        lr=1e-3, 
        weight_decay=0.0, 
        beta=0.,
        n_pow=4,
        data_pts=None,
        n_data_sample=None,
        n_topk=5,
        density_weight=1.,
        euclidean=False,
    ):
        super().__init__(fcn, in_dim, hidden_dim, n_tsteps, lam, lr, weight_decay, beta, n_pow)
        self.save_hyperparameters()
        self.register_buffer("data_pts", data_pts)
        self.n_data_sample = n_data_sample
        self.n_topk = n_topk
        self.density_weight = density_weight

    def density_loss(self, x_t_flat, data_pts):
        vals, inds = torch.topk(
            torch.cdist(x_t_flat, data_pts), k=self.n_topk, dim=-1, largest=False, sorted=False
        )
        return vals.mean()

    def length_loss(self, t, x):
        if self.hparams.euclidean:
            x_flat = x.view(-1, x.shape[2])
            # metric_flat = pullback_metric(x_flat, self.hparams.fcn, create_graph=True, retain_graph=True)
            xdot = self.odefunc(t, x)
            xdot_flat = xdot.view(-1, xdot.shape[2])
            # l_flat = torch.sqrt(torch.einsum('Ni,Nij,Nj->N', xdot_flat, metric_flat, xdot_flat))
            l_flat = torch.sqrt(torch.einsum('Ni,Ni->N', xdot_flat, xdot_flat))
            return l_flat.mean()# * (t[-1] - t[0]) # numerical integration, we set t in [0,1].
        else:
            return super().length_loss(t, x)

    def step(self, batch, batch_idx):
        t = self.t
        x0, x1 = batch
        x_t = self.forward(x0)
        mse_loss = F.mse_loss(x_t[-1], x1)
        if self.pretraining:
            return mse_loss
        mpowerede_loss = 0.
        if self.hparams.beta > 0.:
            mpowerede_loss = (torch.pow(x_t[-1] - x1, self.hparams.n_pow)).mean() * self.hparams.beta
        len_loss = self.length_loss(t, x_t)
        loss = len_loss + self.hparams.lam * mse_loss + mpowerede_loss
        if self.density_weight > 0.:
            x_t_flat = x_t.view(-1, x_t.shape[2])
            if self.n_data_sample is not None and self.n_data_sample < self.data_pts.size(0):
                indices = torch.randperm(self.data_pts.size(0))[:self.n_data_sample]
                dloss = self.density_loss(x_t_flat, self.data_pts[indices])
            else:
                dloss = self.density_loss(x_t_flat, self.data_pts)
            loss += self.density_weight * dloss
        return loss

# DEPRECATED
class GeodesicODEPseudoinv(GeodesicODE):
    def __init__(self, encoder, decoder, in_dim=2, hidden_dim=64, n_tsteps=1000, lam=10, lr=0.001, weight_decay=0, beta=0, n_pow=4):
        super().__init__(encoder, in_dim, hidden_dim, n_tsteps, lam, lr, weight_decay, beta, n_pow)
        self.save_hyperparameters()

    def length_loss(self, t, x):
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[2])
        x_dec_flat = self.hparams.decoder(x_flat)
        metric_flat = pullback_metric(x_dec_flat, self.hparams.fcn, create_graph=True, retain_graph=True, pseudoinverse=True)
        xdot = self.odefunc(t, x)
        xdot_flat = xdot.view(-1, xdot.shape[2])
        l_flat = torch.sqrt(torch.einsum('Ni,Nij,Nj->N', xdot_flat, metric_flat, xdot_flat))
        return l_flat.mean()# * (t[-1] - t[0]) # numerical integration, we set t in [0,1].

def jacobian(func, inputs):
    return compute_jacobian_function(func, inputs)

def velocity(cc, ts, x0, x1):
    '''
    Compute the velocity of the curve at each time point.
    Args:
        cc: Curve module
        ts: torch.Tensor, [T]
        x0: torch.Tensor, [B, D]
        x1: torch.Tensor, [B, D]
    Output:
        velocities: torch.Tensor, [T, B, D]
    '''
    tsc = ts.clone()
    tsc.requires_grad_(True)
    out = cc(x0, x1, tsc)
    orig_shape = out.size()
    out = out.flatten(1,2)

    jacobian = torch.zeros(*out.size()).to(tsc.device)
    jac = torch.zeros(*out.size()).to(tsc.device)
    for i in range(out.size(1)):
        grad_outputs = torch.zeros(*out.size()).to(tsc.device)
        grad_outputs[:, i] = 1.0
        jac[:,i] = grad(outputs=out, inputs=tsc, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
    jac = jac.reshape(*orig_shape)
    
    return jac

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class CondCurve(nn.Module):
    def __init__(self, input_dim, hidden_dim, scale_factor, symmetric, num_layers):
        super(CondCurve, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scale_factor = scale_factor
        self.symmetric = symmetric
        self.num_layers = num_layers
        
        self.mod_x0_x1 = MLP(input_dim=hidden_dim * 2 + 1,
                             hidden_dim=hidden_dim, 
                             output_dim=input_dim, 
                             num_hidden_layers=num_layers)
                             
        self.x0_emb = MLP(input_dim=input_dim, 
                          hidden_dim=hidden_dim, 
                          output_dim=hidden_dim, 
                          num_hidden_layers=num_layers)
                          
        self.x1_emb = MLP(input_dim=input_dim, 
                          hidden_dim=hidden_dim, 
                          output_dim=hidden_dim, 
                          num_hidden_layers=num_layers)
    
    def forward(self, x0, x1, t):
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        
        x0_ = x0.repeat(t.size(0), 1)
        x1_ = x1.repeat(t.size(0), 1)
        t_ = t.repeat(1, x0.size(0)).view(-1, 1)

        emb_x0 = self.x0_emb(x0_)
        emb_x1 = self.x1_emb(x1_)

        avg = t_ * x1_ + (1 - t_) * x0_
        enveloppe = self.scale_factor * (1 - (t_ * 2 - 1).pow(2))
        
        aug_state = torch.cat([emb_x0, emb_x1, t_], dim=-1)

        outs = self.mod_x0_x1(aug_state) * enveloppe + avg

        return outs.view(t.size(0), x0.size(0), self.input_dim)

class GeodesicBridge(pl.LightningModule):
    def __init__(self,
                 func,
                 input_dim,
                 hidden_dim,
                 scale_factor,
                 symmetric,
                 num_layers,
                 lr,
                 weight_decay,
                 n_tsteps=1000):
        super(GeodesicBridge, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scale_factor = scale_factor
        self.symmetric = symmetric
        self.num_layers = num_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.func = func
        self.cc = CondCurve(input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            scale_factor=scale_factor,
                            symmetric=symmetric,
                            num_layers=num_layers)

        self.ts = torch.linspace(0, 1, n_tsteps)
        # self.register_buffer("t", ts)

    def forward(self, x0, x1, t):
        return self.cc(x0, x1, t)

    def length_loss(self, vectors_flat, jac_flat):
        return torch.sqrt((torch.einsum("nij,nj->ni", jac_flat, vectors_flat)**2).sum(axis=1)).mean()
        # loss = torch.sqrt(torch.square(jac_flat @ vectors_flat).sum(axis=1)).mean()

    def step(self, batch, batch_idx):
        x0, x1 = batch
        vectors = velocity(self.cc, self.ts, x0, x1)
        cc_pts = self.cc(x0, x1, self.ts)
        vectors_flat = vectors.flatten(0,1)
        cc_pts_flat = cc_pts.flatten(0, 1)
        jac_flat = jacobian(self.func, cc_pts_flat)
        loss = self.length_loss(vectors_flat, jac_flat)
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
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class GeodesicBridgeDensity(GeodesicBridge):
    def __init__(self,
                 func,
                 input_dim,
                 hidden_dim,
                 scale_factor,
                 symmetric,
                 num_layers,
                 lr,
                 weight_decay,
                 data_pts, # to keep on submfd
                 n_data_sample = None,
                 n_topk = 5,
                 n_tsteps=100,
                 density_weight=1.,
                 euclidean=False):
        super().__init__(
            func=func,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            scale_factor=scale_factor,
            symmetric=symmetric,
            num_layers=num_layers,
            lr=lr,
            weight_decay=weight_decay,
            n_tsteps=n_tsteps
        )
        self.register_buffer("data_pts", data_pts)
        self.n_data_sample = n_data_sample
        self.n_topk = n_topk
        self.density_weight = density_weight
        self.euclidean = euclidean
        if self.euclidean and func is not None:
            warnings.warn("Warning: 'euclidean' flag is set to True, but 'func' is not None. func will not be used.")

        # self.register_buffer("t", ts)

    def density_loss(self, cc_pts_flat, data_pts):
        vals, inds = torch.topk(
            torch.cdist(cc_pts_flat, data_pts), k=self.n_topk, dim=-1, largest=False, sorted=False
        )
        return vals.mean()

    def step(self, batch, batch_idx):
        x0, x1 = batch
        vectors = velocity(self.cc, self.ts, x0, x1)
        cc_pts = self.cc(x0, x1, self.ts)
        vectors_flat = vectors.flatten(0,1)
        cc_pts_flat = cc_pts.flatten(0, 1)
        if self.euclidean:
            loss = torch.sqrt(torch.square(vectors_flat).sum(axis=1)).mean()
        else:
            jac_flat = jacobian(self.func, cc_pts_flat)
            loss = self.length_loss(vectors_flat, jac_flat)
        if self.density_weight > 0.:
            if self.n_data_sample is not None and self.n_data_sample < self.data_pts.size(0):
                indices = torch.randperm(self.data_pts.size(0))[:self.n_data_sample]
                dloss = self.density_loss(cc_pts_flat, self.data_pts[indices])
            else:
                dloss = self.density_loss(cc_pts_flat, self.data_pts)
            loss += self.density_weight * dloss
        return loss

# [DEPRECATED] Use GeodesicBridgeDensity and set euclidean=True.
class GeodesicBridgeDensityEuc(GeodesicBridgeDensity):
    def __init__(self,
                #  func,
                 input_dim,
                 hidden_dim,
                 scale_factor,
                 symmetric,
                 num_layers,
                 lr,
                 weight_decay,
                 data_pts, # to keep on submfd
                 n_data_sample = None,
                 n_topk = 5,
                 n_tsteps=1000,
                 density_weight=1.,
                 normalize_weight=1,):
        super().__init__(
            func=None,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            scale_factor=scale_factor,
            symmetric=symmetric,
            num_layers=num_layers,
            lr=lr,
            weight_decay=weight_decay,
            data_pts=data_pts,
            n_data_sample=n_data_sample,
            n_topk=n_topk,
            n_tsteps=n_tsteps,
            density_weight=density_weight,
        )
        self.normalize_weight=normalize_weight

    def step(self, batch, batch_idx):
        x0, x1 = batch
        vectors = velocity(self.cc, self.ts, x0, x1)
        cc_pts = self.cc(x0, x1, self.ts)
        vectors_flat = vectors.flatten(0,1)
        cc_pts_flat = cc_pts.flatten(0, 1)
        # jac_flat = jacobian(self.func, cc_pts_flat)
        # loss = self.length_loss(vectors_flat, jac_flat)
        loss = torch.sqrt(torch.square(vectors_flat).sum(axis=1)).mean()
        if self.density_weight > 0.:
            if self.n_data_sample is not None and self.n_data_sample < self.data_pts.size(0):
                indices = torch.randperm(self.data_pts.size(0))[:self.n_data_sample]
                dloss = self.density_loss(cc_pts_flat, self.data_pts[indices])
            else:
                dloss = self.density_loss(cc_pts_flat, self.data_pts)
            loss += self.density_weight * dloss
        if self.normalize_weight > 0:
            nloss = torch.square(torch.square(vectors_flat).sum(axis=1) - 1).mean()
            loss += self.normalize_weight * nloss
        return loss
