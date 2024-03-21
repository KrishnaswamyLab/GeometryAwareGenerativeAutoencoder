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

def compute_jacobian_function(f, x, create_graph=True, retain_graph=True):
    """
    Compute the Jacobian of the decoder wrt a batch of points in the latent space using an efficient broadcasting approach.
    :param model: The VAE model.
    :param z_batch: A batch of points in the latent space (tensor).
    :return: A batch of Jacobian matrices.
    """
    # z_batch = z_batch.clone().detach().requires_grad_(True)
    x = x.clone()
    x.requires_grad_(True)
    # model.no_grad()
    output = f(x)
    batch_size, output_dim, latent_dim = *output.shape, x.shape[-1]

    # Use autograd's grad function to get gradients for each output dimension
    jacobian = torch.zeros(batch_size, output_dim, latent_dim).to(x.device)
    for i in range(output_dim):
        grad_outputs = torch.zeros(batch_size, output_dim).to(x.device)
        grad_outputs[:, i] = 1.0
        gradients = grad(outputs=output, inputs=x, grad_outputs=grad_outputs, create_graph=create_graph, retain_graph=retain_graph, only_inputs=True)[0]
        jacobian[:, i, :] = gradients
    return jacobian

def pullback_metric(x, fcn, create_graph=True, retain_graph=True):
    jac = compute_jacobian_function(fcn, x, create_graph, retain_graph)
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
        alpha=0.01,
        beta=0.,
        n_pow=4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.odefunc = ODEFunc(in_dim, hidden_dim)
        self.pretraining = False
        self.t = torch.linspace(0, 1, self.hparams.n_tsteps)


    def length_loss(self, t, x):
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[2])
        metric_flat = pullback_metric(x_flat, self.hparams.fcn, create_graph=False, retain_graph=True)
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
            mpowerede_loss = (torch.pow(x_t[-1] - x1, self.hparams.n_pow) * self.hparams.alpha).mean() * self.hparams.beta
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

