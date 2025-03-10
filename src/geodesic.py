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
import ot as pot
import networkx as nx
import matplotlib.pyplot as plt
from lightning.pytorch.utilities import grad_norm
import shutil
import os

def compute_jacobian_function(f, x, create_graph=True, retain_graph=True):
    """
    Compute the Jacobian of function f wrt x using an efficient broadcasting approach.
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

class GeodesicODE(pl.LightningModule):
    def __init__(self, 
        fcn, # encoder/decoder
        in_dim=2,
        hidden_dim=64, 
        n_tsteps=1000, # num of t steps for length evaluation
        # layer_widths=[64, 64, 64], 
        # activation_fn=torch.nn.ReLU(), 
        lr=1e-3, 
        weight_decay=0.0, 
        # dropout=0.0, 
        # batch_norm=False,
        beta=0.,
        n_pow=4,
        len_weight=1.,
        endpts_weight=1., # regularization for end point
        discriminator_func=None,
        discriminator_weight=0.,
        discriminator_func_for_grad=None,
        discriminator_func_for_grad_weight=1.,
        data_pts=None,
        density_weight=0.0,
        eps = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.odefunc = ODEFunc(in_dim, hidden_dim)
        self.pretraining = False
        self.t = torch.linspace(0, 1, self.hparams.n_tsteps)


    def length_loss(self, vectors_flat, jac_flat):
        return torch.sqrt((torch.einsum("nij,nj->ni", jac_flat, vectors_flat)**2).sum(axis=1)).mean()

    def forward(self, x0):
        t = self.t
        x_t = odeint(self.odefunc, x0, t)
        return x_t

    def step(self, batch, batch_idx):
        t = self.t
        x0, x1 = batch # [B, D]
        x_t = self.forward(x0)

        mse_loss = F.mse_loss(x_t[-1], x1) # endpoint loss
        if self.pretraining:
            return mse_loss
        mpowerede_loss = 0.
        if self.hparams.beta > 0.:
            mpowerede_loss = (torch.pow(x_t[-1] - x1, self.hparams.n_pow)).mean() * self.hparams.beta

        x_flat = x_t.view(-1, x_t.shape[2]) # [T*B, D]
        jac_flat = jacobian(self.hparams.fcn, x_flat) # [T*B, n, D]
        xdot = self.odefunc(t, x_t) # velocity [T, B, D]
        xdot_flat = xdot.view(-1, xdot.shape[2]) # [T*B, D]
        len_loss = self.length_loss(xdot_flat, jac_flat)

        # p(x) Discriminator loss for penalizing the curve to be in the high probability region of the discriminator.
        if self.hparams.discriminator_func is not None and self.hparams.discriminator_weight > 0.:
            # loss = loss + self.discriminator_weight * (1. - self.discriminator_func.positive_proba(cc_pts_flat)).mean()
            disc_loss = (1. - self.hparams.discriminator_func(x_flat)).max()
        
        # dp(x)/dx loss for penalizing the curve to have zero normal component in surface.
        if self.discriminator_func_for_grad is not None and self.discriminator_func_for_grad_weight > 0.:
            disc_jac = jacobian(self.discriminator_func_for_grad, x_flat).squeeze() 
            disc_jac_normalized = (disc_jac + self.eps) / (torch.sqrt(torch.square(disc_jac).sum(axis=1)).reshape(-1,1) + self.eps)
            prj_lengthssq = torch.square((xdot_flat * disc_jac_normalized).sum(axis=1))

            # points penalty to increase the penalty for the points that are start/end points.
            if self.points_penalty is not None and self.points_penalty_grad:
                prj_lengthssq = prj_lengthssq.reshape(xdot_flat.size(0), xdot_flat.size(1)) 
                prj_lengthssq = prj_lengthssq * self.points_penalty.reshape(-1, 1)
                prj_lengthssq = prj_lengthssq.flatten()

            prj_loss = torch.square(prj_lengthssq).mean()
            # prj_loss = torch.square(prj_lengthssq).max() # using max.
        
        # density loss
        if self.data_pts is not None and self.density_weight > 0.:
            if self.n_data_sample is not None and self.n_data_sample < self.data_pts.size(0):
                indices = torch.randperm(self.data_pts.size(0))[:self.n_data_sample]
                dloss = self.density_loss(x_flat, self.data_pts[indices], x_flat)
            else:
                dloss = self.density_loss(x_flat, self.data_pts, x_flat)

        loss = len_loss * self.hparams.len_weight + self.hparams.endpts_weight * mse_loss + mpowerede_loss \
            + self.hparams.discriminator_weight * disc_loss \
            + self.discriminator_func_for_grad_weight * prj_loss + self.hparams.density_weight * dloss
        
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

        #self.init_layers()
    
    def forward(self, x):
        return self.layers(x)
    
    def init_layers(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

class CondCurve(nn.Module):
    def __init__(self, input_dim, hidden_dim, scale_factor, symmetric, num_layers, k=2, embed_t=True,
                 init_method='line', graph_pts=None, graph_pts_encodings=None, encoder=None, 
                 diff_op=None, diff_t=1):
        super(CondCurve, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scale_factor = scale_factor
        self.symmetric = symmetric
        self.num_layers = num_layers
        self.k = k
        self.embed_t = embed_t
        self.init_method = init_method
        self.graph_pts = graph_pts
        self.graph_pts_encodings = graph_pts_encodings
        self.initial_curve = None
        self.encoder = encoder
        if self.embed_t:
            self.aug_dim = 3 * hidden_dim
        else:
            self.aug_dim = 2 * hidden_dim + 1
        self.mod_x0_x1 = MLP(input_dim=self.aug_dim,
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
        
        if self.embed_t:
            self.t_emb = MLP(input_dim=1,
                            hidden_dim=hidden_dim, 
                            output_dim=hidden_dim, 
                            num_hidden_layers=num_layers)
        self.diff_op = diff_op
        self.diff_t = diff_t
        if self.init_method == 'djikstra' or self.init_method == 'diffusion':
            self.graph_pts = graph_pts
            self.graph_pts_encodings = graph_pts_encodings
            self.G = self._construct_graph(self.graph_pts_encodings, method=self.init_method, knn=None,
                                           diff_op=self.diff_op, diff_t=self.diff_t)
            print("Graph constructed with", self.G.number_of_nodes(), "nodes and", self.G.number_of_edges(), "edges")
            #print(list(self.G.nodes))
    def _random_walk(self, diff_op, source_idx, target_idx, topk=1):
        '''
        Perform random walk from source to target.
        diff_op: torch.Tensor, [N, N], normalized transition probabilities per row.
        source_idx: int, index of the source node
        target_idx: int, index of the target node
        topk: int, number of candidates to consider at each step
        '''
        G = self.G
        path = []
        current_idx = source_idx
        path.append(current_idx)
        while current_idx != target_idx:
            if topk is not None:
                candidates = np.argsort(-diff_op[current_idx])[:topk]
                cand_probs = diff_op[current_idx][candidates]
                cand_probs = cand_probs / cand_probs.sum()
            else:
                candidates = np.arange(diff_op.shape[1])
                cand_probs = diff_op[current_idx]
            # choose a random candidate according to the diffusion probabilities.
            candidate = np.random.choice(candidates, p=cand_probs, size=1)
            current_idx = candidate[0]
            path.append(current_idx)
        return path
    
    def _construct_graph(self, graph_pts, method='djikstra', knn=5, diff_op=None, diff_t=1):
            # construct a graph with nodes at graph_pts of [N, D]
            N = graph_pts.shape[0]
            G = nx.Graph()
            for i in range(N):
                G.add_node(int(i))
            
            # Edges
            if method == 'djikstra':
                dist_mat = torch.cdist(graph_pts, graph_pts) # [N, N]
            elif method == 'diffusion':
                assert diff_op is not None, "Diffusion operator is not provided."
                diff_op = torch.tensor(diff_op)
                if diff_t > 1:
                    powered_diff_op = torch.matrix_power(diff_op, torch.tensor(diff_t, dtype=torch.int32))
                else:
                    powered_diff_op = diff_op
                dist_mat = 1 - powered_diff_op
            print("Graph method:", method, "knn:", knn, "dist_mat.shape:", dist_mat.shape,
                  "diff_t:", diff_t)
            if knn is not None:
                vals, inds = torch.topk(dist_mat, k=knn, dim=-1, largest=False, sorted=False) # [N, knn]
                # Add topknn edges
                for i in range(N):
                    for j in inds[i]:
                        if i != j:
                            G.add_edge(int(i), int(j), weight=float(dist_mat[i,j].cpu().numpy()))
            else: # add all edges
                for i in range(N):
                    avg_dist = dist_mat[i].mean()
                    for j in range(N):
                        if i != j:
                            if dist_mat[i,j] <= 0.99:
                                #G.add_edge(int(i), int(j), weight=float(dist_mat[i,j].cpu().numpy()))
                                G.add_edge(int(i), int(j), weight=-1)

            print("Graph is fully connected? ", nx.is_connected(G))
            return G
    
    def init_curve(self, x0, x1, t, num_steps, method='line', graph_pts=None, graph_pts_encodings=None, diff_op=None):
        '''
        Initialize the curve using the method.
        Args:
            x0: torch.Tensor, [T*B, D]
            x1: torch.Tensor, [T*B, D]
            t: torch.Tensor, [T*B, 1]
            num_steps: int, number of steps in the curve, aka T.
            method: str, 'line' or 'djikstra'
            graph_pts: torch.Tensor [N, D] of graph points
            graph_pts_encodings: torch.Tensor [N, E] of graph points encodings
        Output:
            curve: torch.Tensor, [T*B, D]
        '''
        if method == 'line':
            #print("Using straight line to initialize the curve...")
            return (1-t) * x0 + t * x1
        elif method == 'djikstra' or method == 'diffusion' and graph_pts is not None:
            print(f"Using {method} to initialize the curve...")
            # Find shortest paths from x0 to x1 using Dijkstra's algorithm
            G = self.G
            #print("Graph nodes:", list(G.nodes))
            #import pdb; pdb.set_trace();
            _x0 = x0.view(num_steps, -1, self.input_dim)[0, :, :] # [B, D]
            _x1 = x1.view(num_steps, -1, self.input_dim)[0, :, :] # [B, D]
            _z0 = self.encoder(_x0)
            _z1 = self.encoder(_x1)
            # Find the index of the closest point in graph_pts to x0 and x1
            _x0_idx = torch.argmin(torch.cdist(_x0, graph_pts), dim=-1).cpu().numpy().astype(int) # [B]
            _x1_idx = torch.argmin(torch.cdist(_x1, graph_pts), dim=-1).cpu().numpy().astype(int) # [B]
            shortest_paths = [[] for _ in range(_x0.shape[0])]

            path_method = 'random_walk'
            for i in range(_x0_idx.shape[0]):
                if path_method == 'random_walk':
                    shortest_path = self._random_walk(diff_op, int(_x0_idx[i]), int(_x1_idx[i]), topk=40)
                else:
                    shortest_path = nx.shortest_path(G, source=int(_x0_idx[i]), target=int(_x1_idx[i]), weight='weight')                
                shortest_paths[i] = shortest_path

            # Take num_steps points from the shortest path
            curve_pts = []
            print('x0.shape[0]:', _x0.shape[0])
            for i in range(_x0.shape[0]):
                print(f"Shortest path for curve {i}: Length {len(shortest_paths[i])}")
                if num_steps > len(shortest_paths[i]):
                    raise ValueError(f"num_steps is larger than the shortest path length: {num_steps} > {len(shortest_paths[i])}")
                t_idxs = torch.linspace(0, len(shortest_paths[i])-1, num_steps)  # [num_steps]
                t_idxs = t_idxs.cpu().numpy().astype(int)
                path = graph_pts[np.array(shortest_paths[i])[t_idxs]]
                print("Sampled path:", np.array(shortest_paths[i])[t_idxs], "path:", path.shape)
                curve_pts.append(path) # [num_steps, D]
            
            #import pdb; pdb.set_trace();
            curve_pts = torch.stack(curve_pts) # [B, num_steps, D]
            curve_pts = torch.transpose(curve_pts, 0, 1) # [num_steps, B, D]

            # Visualize matched x1, x0 and the curve
            ax = plt.axes(projection='3d')
            ax.scatter(graph_pts_encodings[:,0].cpu().numpy(), graph_pts_encodings[:,1].cpu().numpy(), graph_pts_encodings[:,2].cpu().numpy(), c='gray', label='graph_pts', alpha=0.8)
            ax.scatter(graph_pts_encodings[_x0_idx][:,0].cpu().numpy(), graph_pts_encodings[_x0_idx][:,1].cpu().numpy(), graph_pts_encodings[_x0_idx][:,2].cpu().numpy(), c='blue', label='x0')
            ax.scatter(graph_pts_encodings[_x1_idx][:,0].cpu().numpy(), graph_pts_encodings[_x1_idx][:,1].cpu().numpy(), graph_pts_encodings[_x1_idx][:,2].cpu().numpy(), c='red', label='x1')
            curve_pts_flat = curve_pts.flatten(0,1)
            curve_pts_flat_enc = self.encoder(curve_pts_flat)
            ax.scatter(curve_pts_flat_enc[:,0].cpu().numpy(), curve_pts_flat_enc[:,1].cpu().numpy(), curve_pts_flat_enc[:,2].cpu().numpy(), c='green', label='curve')
            ax.legend()
            plt.savefig("./curve_init.png")

            # plotly
            import plotly.graph_objects as go
            curve_pts_enc = curve_pts_flat_enc.view(num_steps, -1, graph_pts_encodings.shape[-1])
            fig = go.Figure()
            encoded_graph_pts = self.encoder(graph_pts)
            fig.add_trace(go.Scatter3d(x=encoded_graph_pts[:,0].cpu().numpy(), y=encoded_graph_pts[:,1].cpu().numpy(), z=encoded_graph_pts[:,2].cpu().numpy(), mode='markers', name='graph_pts', marker=dict(color='gray')))
            fig.add_trace(go.Scatter3d(x=encoded_graph_pts[_x0_idx][:,0].cpu().numpy(), y=encoded_graph_pts[_x0_idx][:,1].cpu().numpy(), z=encoded_graph_pts[_x0_idx][:,2].cpu().numpy(), mode='markers', name='x0', marker=dict(color='blue')))
            fig.add_trace(go.Scatter3d(x=encoded_graph_pts[_x1_idx][:,0].cpu().numpy(), y=encoded_graph_pts[_x1_idx][:,1].cpu().numpy(), z=encoded_graph_pts[_x1_idx][:,2].cpu().numpy(), mode='markers', name='x1', marker=dict(color='red')))
            for i in range(10):
                fig.add_trace(go.Scatter3d(x=curve_pts_enc[:,i,0].cpu().numpy(), y=curve_pts_enc[:,i,1].cpu().numpy(), z=curve_pts_enc[:,i,2].cpu().numpy(), mode='markers', name='curve'+str(i), marker=dict(color='green', size=5)))
                fig.add_trace(go.Scatter3d(x=curve_pts_enc[:,i,0].cpu().numpy(), y=curve_pts_enc[:,i,1].cpu().numpy(), z=curve_pts_enc[:,i,2].cpu().numpy(), mode='lines', name='curve'+str(i), marker=dict(color='green')))
            fig.write_html("./curve_init.html")

            # Visualize the diffusion operator on the encodings.
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111, projection='3d')
            diff_op = diff_op
            color = diff_op[_x0_idx[0]]
            pc = ax.scatter(graph_pts_encodings[:,0].cpu().numpy(), graph_pts_encodings[:,1].cpu().numpy(), graph_pts_encodings[:,2].cpu().numpy(), 
                       c=color, cmap='viridis')
            plt.colorbar(pc, label=f'Diffusion Prob at pt at idx {_x0_idx[0]}')
            plt.savefig("./diffusion_operator.png")

            # plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=encoded_graph_pts[_x0_idx][:,0].cpu().numpy(), y=encoded_graph_pts[_x0_idx][:,1].cpu().numpy(), z=encoded_graph_pts[_x0_idx][:,2].cpu().numpy(), 
                                       mode='markers', name='x0', marker=dict(color='gray', line=dict(color='black', width=2)), opacity=1.0))
            fig.add_trace(go.Scatter3d(x=encoded_graph_pts[:,0].cpu().numpy(), y=encoded_graph_pts[:,1].cpu().numpy(), z=encoded_graph_pts[:,2].cpu().numpy(), 
                                       mode='markers', name='graph_pts', marker=dict(color=color), opacity=0.8,
                                       hovertext=color))
            fig.write_html("./diffusion_operator.html")

            return curve_pts.flatten(0,1) # [num_steps*B, D]
        else:
            raise ValueError(f"Unknown initialization method: {method}")
        

    
    def forward(self, x0, x1, t):
        '''
        Args:
            x0: torch.Tensor, [B, D]
            x1: torch.Tensor, [B, D]
            t: torch.Tensor, [T, 1]
        Output:
            curve: torch.Tensor, [T, B, D]
        '''
        t = t.unsqueeze(-1) if t.dim() == 1 else t # [T, 1]
        
        x0_ = x0.repeat(t.size(0), 1) # [T*B, D]
        x1_ = x1.repeat(t.size(0), 1) 
        num_steps = t.size(0)
        t_ = t.repeat(1, x0.size(0)).view(-1, 1) # [T*B, 1] 

        emb_x0 = self.x0_emb(x0_)
        emb_x1 = self.x1_emb(x1_)
        if self.embed_t:
            emb_t = self.t_emb(t_) # [T*B, D]
        
        # if self.initial_curve is None:
        #     print("Initializing the curve...", 'x0', x0_.shape, 'x1', x1_.shape, 't', t_.shape, 'num_steps', num_steps)
        #     self.initial_curve = self.init_curve(x0_, x1_, t_, num_steps, self.init_method, self.graph_pts, self.graph_pts_encodings)
        self.initial_curve = self.init_curve(x0_, x1_, t_, num_steps, self.init_method, self.graph_pts, self.graph_pts_encodings, self.diff_op)

        enveloppe = self.scale_factor * (1 - (t_ * 2 - 1).pow(self.k)) # [T*B, 1]
        
        if self.embed_t:
            aug_state = torch.cat([emb_x0, emb_x1, emb_t], dim=-1) # [T*B, 3*D]
        else:
            aug_state = torch.cat([emb_x0, emb_x1, t_], dim=-1) # [T*B, 2*D+1]

        outs = self.mod_x0_x1(aug_state) * enveloppe + self.initial_curve

        return outs.view(t.size(0), x0.size(0), self.input_dim)

class CondCurveOverfit(CondCurve):
    def __init__(self, input_dim, hidden_dim, scale_factor, symmetric, num_layers, id_dim, id_emb_dim, k=2, embed_t=False,
                 init_method='line', graph_pts=None, graph_pts_encodings=None, encoder=None,
                 diff_op=None, diff_t=1.0):
        super(CondCurveOverfit, self).__init__(input_dim, hidden_dim, scale_factor, symmetric, num_layers, k, embed_t,
                                               init_method, graph_pts, graph_pts_encodings, encoder,
                                               diff_op, diff_t)
        self.id_net = MLP(input_dim=id_dim,
                          hidden_dim=id_emb_dim,
                          output_dim=id_emb_dim,
                          num_hidden_layers=1)
        
        if self.embed_t:
            self.aug_dim = 2 * hidden_dim + hidden_dim + id_emb_dim
        else:
            self.aug_dim = 2 * hidden_dim + 1 + id_emb_dim

        self.mod_x0_x1 = MLP(input_dim=self.aug_dim,
                        hidden_dim=hidden_dim, 
                        output_dim=input_dim, 
                        num_hidden_layers=num_layers)
    
    def forward(self, x0, x1, t, ids):
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        
        x0_ = x0.repeat(t.size(0), 1) # [T*B, D]
        x1_ = x1.repeat(t.size(0), 1)
        num_steps = t.size(0)
        ids_ = ids.repeat(t.size(0), 1) 
        t_ = t.repeat(1, x0.size(0)).view(-1, 1) # [T*B, 1]

        emb_x0 = self.x0_emb(x0_)
        emb_x1 = self.x1_emb(x1_)
        if self.embed_t:
            emb_t = self.t_emb(t_) # [T*B, D]
        # if self.initial_curve is None:
        #     import pdb; pdb.set_trace();
        #     print("Initializing the curve...", 'x0', x0_.shape, 'x1', x1_.shape, 't', t_.shape, 'num_steps', num_steps)
        #     self.initial_curve = self.init_curve(x0_, x1_, t_, num_steps, self.init_method, self.graph_pts, self.graph_pts_encodings)
        self.initial_curve = self.init_curve(x0_, x1_, t_, num_steps, 
                                             self.init_method, self.graph_pts, self.graph_pts_encodings, self.diff_op)

        enveloppe = self.scale_factor * (1 - (t_ * 2 - 1).pow(self.k))
        
        ids_emb = self.id_net(ids_)
        if self.embed_t:
            aug_state = torch.cat([emb_x0, emb_x1, emb_t, ids_emb], dim=-1) # [T*B, 3*D+id_dim]
        else:
            aug_state = torch.cat([emb_x0, emb_x1, t_, ids_emb], dim=-1) # [T*B, 2*D+1+id_dim]
        
        outs = self.mod_x0_x1(aug_state) * enveloppe + self.initial_curve

        return outs.view(t.size(0), x0.size(0), self.input_dim)

class GeodesicBridge(pl.LightningModule):
    def __init__(self,
                 func,
                 input_dim,
                 hidden_dim,
                 scale_factor,
                 symmetric,
                 embed_t,
                 num_layers,
                 lr,
                 weight_decay,
                 n_tsteps=1000,
                 discriminator_func=None,
                 discriminator_weight=0.,
                 multiply_loss=False,
                 discriminator_func_for_grad=None,
                 discriminator_func_for_grad_weight=1.,
                 eps = 1e-5,
                 length_weight=1.,
                 data_pts=None,
                 n_data_sample = None,
                 n_topk = 5,
                 n_top_k_sel=0,
                density_weight=1.,
                points_penalty_alpha=0.,
                disc_use_max=False,
                points_penalty_power=2,
                points_penalty_grad=False,
                points_penalty_disc=False,
                points_penalty_density=False,
                cc_k=2,
                init_method='line',
                graph_pts=None,
                graph_pts_encodings=None,
                encoder=None,
                diff_op=None,
                diff_t=1.0,
                ):
        #super().__init__()
        super(GeodesicBridge, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scale_factor = scale_factor
        self.symmetric = symmetric
        self.num_layers = num_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.func = func

        ts = torch.linspace(0, 1, n_tsteps)
        self.register_buffer("ts", ts)
        self.discriminator_func = discriminator_func
        self.discriminator_weight = discriminator_weight
        self.multiply_loss = multiply_loss
        self.discriminator_func_for_grad = discriminator_func_for_grad
        self.discriminator_func_for_grad_weight = discriminator_func_for_grad_weight
        self.length_weight = length_weight
        self.eps = eps
        self.density_weight = density_weight
        self.register_buffer("data_pts", data_pts)
        self.n_data_sample = n_data_sample
        self.n_topk = n_topk
        self.n_top_k_sel = n_top_k_sel
        self.points_penalty_alpha = points_penalty_alpha
        self.points_penalty = None
        if points_penalty_alpha > 0.:
            self.points_penalty = points_penalty_alpha * (self.ts - self.ts.mean())**points_penalty_power + 1
            self.points_penalty = self.points_penalty / self.points_penalty.sum()
            # self.points_penalty = points_penalty_alpha * (self.ts - self.ts.mean())**points_penalty_power
        self.disc_use_max = disc_use_max
        self.points_penalty_grad = points_penalty_grad
        self.points_penalty_disc = points_penalty_disc
        self.points_penalty_density = points_penalty_density
        # self.register_buffer("t", ts)
        # if self.func is not None:
        #     for param in self.func.parameters():
        #         param.requires_grad = False
        # Freeze self.discriminator if it is not None
        # if self.discriminator is not None:
        #     for param in self.discriminator.parameters():
        #         param.requires_grad = False

        self.cc = CondCurve(input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    scale_factor=scale_factor,
                    symmetric=symmetric,
                    embed_t=embed_t,
                    num_layers=num_layers,
                    k=cc_k,
                    init_method=init_method,
                    graph_pts=graph_pts,
                    graph_pts_encodings=graph_pts_encodings,
                    encoder=encoder,
                    diff_op=diff_op,
                    diff_t=diff_t,
                )

    def forward(self, x0, x1, t):
        return self.cc(x0, x1, t)

    def length_loss(self, vectors_flat, jac_flat):
        return torch.sqrt((torch.einsum("nij,nj->ni", jac_flat, vectors_flat)**2).sum(axis=1)).mean()
        # loss = torch.sqrt(torch.square(jac_flat @ vectors_flat).sum(axis=1)).mean()

    def density_loss(self, cc_pts_flat, data_pts, cc_pts):
        vals, inds = torch.topk(
            torch.cdist(cc_pts_flat, data_pts), k=self.n_topk, dim=-1, largest=False, sorted=False
        )
        if self.n_top_k_sel > 0:
            assert self.n_top_k_sel <= self.n_topk
            # randomly select without put back
            inds = torch.randperm(vals.size(0))[:self.n_top_k_sel]
            vals = vals[inds]
        if self.points_penalty is not None and self.points_penalty_density:
            vals = vals.reshape(cc_pts.size(0), -1)
            vals = vals * self.points_penalty.reshape(-1,1)
        hinge = 0.5
        vals = vals - hinge
        vals[vals < 0] = 0 
        return vals.mean()

    def step(self, batch, batch_idx):
        x0, x1 = batch
        vectors = velocity(self.cc, self.ts, x0, x1)
        cc_pts = self.cc(x0, x1, self.ts)
        vectors_flat = vectors.flatten(0,1)
        cc_pts_flat = cc_pts.flatten(0, 1)
        jac_flat = jacobian(self.func, cc_pts_flat)
        len_loss = self.length_loss(vectors_flat, jac_flat)
        loss = self.length_weight * len_loss
        if self.discriminator_func is not None and self.discriminator_weight > 0.:
            # loss = loss + self.discriminator_weight * (1. - self.discriminator_func.positive_proba(cc_pts_flat)).mean()
            disc_val = self.discriminator_func(cc_pts_flat)
            if self.disc_use_max:
                disc_loss = (1. - disc_val).max()
            else:
                if self.points_penalty is not None and self.points_penalty_disc:
                    disc_val = disc_val.reshape(cc_pts.size(0),cc_pts.size(1))
                    disc_val = disc_val * self.points_penalty.reshape(-1,1)
                    disc_val = disc_val.flatten()
                disc_loss = (1. - disc_val).mean()
            # disc_loss = (1. - self.discriminator_func(cc_pts_flat)).max() # using max.
            # print(disc_loss.item())
            # loss = loss + self.discriminator_weight * disc_loss * loss # increase the penalty for longer curves.
            # loss = loss + self.discriminator_weight * disc_loss
            if self.multiply_loss:
                disc_loss = disc_loss * len_loss # increase the penalty for longer curves.
            loss = loss + self.discriminator_weight * disc_loss
        if self.discriminator_func_for_grad is not None and self.discriminator_func_for_grad_weight > 0.:
            disc_jac = jacobian(self.discriminator_func_for_grad, cc_pts_flat).squeeze()
            disc_jac_normalized = (disc_jac + self.eps) / (torch.sqrt(torch.square(disc_jac).sum(axis=1)).reshape(-1,1) + self.eps)
            prj_lengthssq = torch.square((vectors_flat * disc_jac_normalized).sum(axis=1))
            if self.points_penalty is not None and self.points_penalty_grad:
                prj_lengthssq = prj_lengthssq.reshape(vectors.size(0), vectors.size(1))
                prj_lengthssq = prj_lengthssq * self.points_penalty.reshape(-1, 1)
                prj_lengthssq = prj_lengthssq.flatten()
            # prj_loss = torch.square(prj_lengthssq).mean() # [FIXME] I accidentally squared twice!
            prj_loss = prj_lengthssq.mean() # fixed.
            # prj_loss = torch.square(prj_lengthssq).max() # using max.
            if self.multiply_loss:
                prj_loss = prj_loss * len_loss # increase the penalty for longer curves.
            loss = loss + self.discriminator_func_for_grad_weight * prj_loss
        if self.data_pts is not None and self.density_weight > 0.:
            if self.n_data_sample is not None and self.n_data_sample < self.data_pts.size(0):
                indices = torch.randperm(self.data_pts.size(0))[:self.n_data_sample]
                dloss = self.density_loss(cc_pts_flat, self.data_pts[indices], cc_pts)
            else:
                dloss = self.density_loss(cc_pts_flat, self.data_pts, cc_pts)
            loss = loss + self.density_weight * dloss

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

    # def configure_optimizers(self):
    #     # Collect all parameters of the model that are not part of the discriminator
    #     if self.discriminator is not None:
    #         # Create a set of all discriminator parameter ids
    #         discriminator_params = set(p for p in self.discriminator.parameters())
    #         # Filter model parameters to exclude those that are part of the discriminator
    #         params = [p for p in self.parameters() if p not in discriminator_params]
    #     else:
    #         # If no discriminator, just use all parameters
    #         params = self.parameters()

    #     # Create and return the optimizer with only the desired parameters
    #     optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
    #     return optimizer

class GeodesicBridgeOverfit(GeodesicBridge):
    def __init__(self,
                 func,
                 input_dim,
                 hidden_dim,
                 scale_factor,
                 symmetric,
                 embed_t,
                 num_layers,
                 lr,
                 weight_decay,
                 n_tsteps=1000,
                 discriminator_func=None,
                 discriminator_weight=0.,
                 multiply_loss=False,
                 discriminator_func_for_grad=None,
                 discriminator_func_for_grad_weight=1.,
                 eps = 1e-5,
                 length_weight=1.,
                 data_pts=None,
                 n_data_sample = None,
                 n_topk = 5,
                 n_top_k_sel=0,
                density_weight=1.,
                points_penalty_alpha=0.,
                disc_use_max=False,
                points_penalty_power=2,
                points_penalty_grad=False,
                points_penalty_disc=False,
                points_penalty_density=False,
                id_dim=0,  # learn an embedding for each curve, to overfit the model to each curve.
                id_emb_dim=0,
                cc_k=2,
                init_method='line',
                graph_pts=None,
                graph_pts_encodings=None,
                encoder=None,
                diff_op=None,
                diff_t=1.0,
                ):
        super().__init__(
            func=func,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            scale_factor=scale_factor,
            symmetric=symmetric,
            embed_t=embed_t,
            num_layers=num_layers,
            lr=lr,
            weight_decay=weight_decay,
            n_tsteps=n_tsteps,
            discriminator_func=discriminator_func,
            discriminator_weight=discriminator_weight,
            multiply_loss=multiply_loss,
            discriminator_func_for_grad=discriminator_func_for_grad,
            discriminator_func_for_grad_weight=discriminator_func_for_grad_weight,
            eps=eps,
            length_weight=length_weight,
            data_pts=data_pts,
            n_data_sample=n_data_sample,
            n_topk=n_topk,
            n_top_k_sel=n_top_k_sel,
            density_weight=density_weight,
            points_penalty_alpha=points_penalty_alpha,
            disc_use_max=disc_use_max,
            points_penalty_power=points_penalty_power,
            points_penalty_grad=points_penalty_grad,
            points_penalty_disc=points_penalty_disc,
            points_penalty_density=points_penalty_density,
            cc_k=cc_k,
            init_method=init_method,
            graph_pts=graph_pts,
            graph_pts_encodings=graph_pts_encodings,
            encoder=encoder,
            diff_op=diff_op,
            diff_t=diff_t,
        )
        assert id_dim>0 and id_emb_dim>0
        self.cc = CondCurveOverfit(input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            scale_factor=scale_factor,
                            symmetric=symmetric,
                            embed_t=embed_t,
                            num_layers=num_layers,
                            id_dim=id_dim,
                            id_emb_dim=id_emb_dim,
                            k=cc_k,
                            init_method=init_method,
                            graph_pts=graph_pts,
                            graph_pts_encodings=graph_pts_encodings,
                            encoder=encoder,
                            diff_op=diff_op,
                            diff_t=diff_t,
                        )
    def forward(self, x0, x1, t, ids):
        return self.cc(x0, x1, t, ids)

    def step(self, batch, batch_idx):
        x0, x1, ids = batch

        def cc_func(x0, x1, t):
            return self.cc(x0, x1, t, ids)

        vectors = velocity(cc_func, self.ts, x0, x1)
        cc_pts = self.cc(x0, x1, self.ts, ids)
        vectors_flat = vectors.flatten(0,1)
        cc_pts_flat = cc_pts.flatten(0, 1)
        jac_flat = jacobian(self.func, cc_pts_flat)
        
        len_loss = self.length_loss(vectors_flat, jac_flat)
        self.log(f'loss_length', len_loss, prog_bar=True, on_epoch=True)
        
        loss = self.length_weight * len_loss

        if self.discriminator_func is not None and self.discriminator_weight > 0.:
            # loss = loss + self.discriminator_weight * (1. - self.discriminator_func.positive_proba(cc_pts_flat)).mean()
            disc_val = self.discriminator_func(cc_pts_flat)
            if self.disc_use_max:
                
                disc_loss = (1. - disc_val).max()
            else:
                if self.points_penalty is not None and self.points_penalty_disc:
                    disc_val = disc_val.reshape(cc_pts.size(0),cc_pts.size(1))
                    disc_val = disc_val * self.points_penalty.reshape(-1,1)
                    disc_val = disc_val.flatten()
                disc_loss = (1. - disc_val).mean()
            # disc_loss = (1. - self.discriminator_func(cc_pts_flat)).max() # using max.
            # print(disc_loss.item())
            # loss = loss + self.discriminator_weight * disc_loss * loss # increase the penalty for longer curves.
            # loss = loss + self.discriminator_weight * disc_loss
            if self.multiply_loss:
                disc_loss = disc_loss * len_loss # increase the penalty for longer curves.
            self.log(f'loss_discriminator', disc_loss, prog_bar=True, on_epoch=True)
            loss = loss + self.discriminator_weight * disc_loss
        if self.discriminator_func_for_grad is not None and self.discriminator_func_for_grad_weight > 0.:
            disc_jac = jacobian(self.discriminator_func_for_grad, cc_pts_flat).squeeze()
            disc_jac_normalized = (disc_jac + self.eps) / (torch.sqrt(torch.square(disc_jac).sum(axis=1)).reshape(-1,1) + self.eps)
            prj_lengthssq = torch.square((vectors_flat * disc_jac_normalized).sum(axis=1))
            if self.points_penalty is not None and self.points_penalty_grad:
                prj_lengthssq = prj_lengthssq.reshape(vectors.size(0), vectors.size(1))
                prj_lengthssq = prj_lengthssq * self.points_penalty.reshape(-1, 1)
                prj_lengthssq = prj_lengthssq.flatten()
            # prj_loss = torch.square(prj_lengthssq).mean() # [FIXME] I accidentally squared twice!
            prj_loss = prj_lengthssq.mean() # fixed.
            # prj_loss = torch.square(prj_lengthssq).max() # using max.
            if self.multiply_loss:
                prj_loss = prj_loss * len_loss # increase the penalty for longer curves.
            self.log(f'loss_discriminator_gradient', prj_loss, prog_bar=True, on_epoch=True)
            loss = loss + self.discriminator_func_for_grad_weight * prj_loss
        if self.data_pts is not None and self.density_weight > 0.:
            if self.n_data_sample is not None and self.n_data_sample < self.data_pts.size(0):
                indices = torch.randperm(self.data_pts.size(0))[:self.n_data_sample]
                dloss = self.density_loss(cc_pts_flat, self.data_pts[indices], cc_pts)
            else:
                dloss = self.density_loss(cc_pts_flat, self.data_pts, cc_pts)
            self.log(f'loss_density', dloss, prog_bar=True, on_epoch=True)
            loss = loss + self.density_weight * dloss
        self.log(f'loss', loss, prog_bar=True, on_epoch=True)
        return loss


"""
[DEPRECATED. MEGRED TO GeodesicBridge.]
"""
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
            loss = loss + self.density_weight * dloss
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
            loss = loss + self.density_weight * dloss
        if self.normalize_weight > 0:
            nloss = torch.square(torch.square(vectors_flat).sum(axis=1) - 1).mean()
            loss = loss + self.normalize_weight * nloss
        return loss

import plotly.graph_objects as go
import os
# import moviepy.editor as mpy
class GeodesicFM(GeodesicBridgeOverfit):
    def __init__(self,
                func,
                encoder,
                input_dim,
                hidden_dim,
                scale_factor=1,
                symmetric=True,
                embed_t=False,
                num_layers=3, 
                n_tsteps=100,
                lr=1e-3,
                weight_decay=1e-3,
                flow_weight=1.,
                length_weight=1.,
                cc_k=2,
                use_density=False,
                data_pts=None,
                data_pts_encodings=None,
                diff_op=None,
                diff_t=1.0,
                density_weight=1.,
                fixed_pot=False, # Whether to fix the OT plan or resample each forward pass.
                init_method='line',
                visualize_training=False,
                dataloader=None,
                device=None,
                training_save_dir='./eb_fm/training/',
                ):
            super().__init__(
                func=func,
                input_dim=input_dim, 
                hidden_dim=hidden_dim, 
                scale_factor=scale_factor, 
                symmetric=symmetric, 
                embed_t=embed_t,
                num_layers=num_layers, 
                n_tsteps=n_tsteps, 
                lr=lr, 
                weight_decay=weight_decay,
                discriminator_weight=0.,
                discriminator_func_for_grad_weight=0.,
                id_dim=1,
                id_emb_dim=1,
                init_method=init_method,
                graph_pts=data_pts,
                graph_pts_encodings=data_pts_encodings,
                diff_op=diff_op,
                diff_t=diff_t,
                length_weight=length_weight,
                cc_k=cc_k,
                data_pts=data_pts,
                density_weight=density_weight,
                encoder=encoder,
            )
            self.encoder = encoder
            self.flow_model = MLP(input_dim=input_dim+1,hidden_dim=hidden_dim,output_dim=input_dim,num_hidden_layers=num_layers)
            self.flow_weight = flow_weight
            self.use_density = use_density
            self.visualize_training = visualize_training
            self.training_save_dir = training_save_dir
            self.data_pts = data_pts # 1) for density loss. 2) for visualization.
            self.dataloader = dataloader # for traj visualization along training.
            self.fixed_pot = fixed_pot

            if self.visualize_training and self.dataloader is not None and self.data_pts is not None:
                visualize_x0 = []
                visualize_x1 = []
                for x0_, x1_ in self.dataloader:
                    visualize_x0.append(x0_)
                    visualize_x1.append(x1_)
                self.visualize_x0 = torch.cat(visualize_x0, dim=0).to(device)
                self.visualize_x1 = torch.cat(visualize_x1, dim=0).to(device)
                self.data_pts_encodings = self.encoder(self.data_pts).cpu().numpy()
            
            if self.fixed_pot:
                self.fixed_pairs = None # Tuple of (i,j), each of shape [B].
            
            # remove old grad norms.
            if os.path.exists('./eb_fm/grad_norms'):
                shutil.rmtree('./eb_fm/grad_norms')
            os.makedirs('./eb_fm/grad_norms')
    
    def sample_optimal_pairs(self, x0, x1, encoder):
        # Optimal transport pair based on latent encodings.
        #x0 = x0[torch.randperm(x0.shape[0])] # [B, D]
        a, b = pot.unif(x0.size()[0]), pot.unif(x1.size()[0]) # [B]
        z0, z1 = encoder(x0), encoder(x1) # [B, d]
        M = torch.cdist(z0, z1) ** 2 # [B, B]
        M = M / M.max()
        pi = pot.emd(a, b, M.detach().cpu().numpy()) # [B, B]
        p = pi.flatten() # [B*B]
        p = p / p.sum()
        choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=x0.shape[0]) # [B]
        i, j = np.divmod(choices, pi.shape[1]) # [B]
        return i, j

    def step(self, batch, batch_idx):
        x0, x1 = batch # [B, D]
        ids = torch.zeros((x0.size(0),1), device=x0.device, dtype=x0.dtype)
        # if self.fixed_pot is False:
        #     i, j = self.sample_optimal_pairs(x0, x1, self.encoder)
        # else:
        #     if self.current_epoch == 0:
        #         i, j = self.sample_optimal_pairs(x0, x1, self.encoder)
        #         self.fixed_pairs = (i, j)
        #     else:
        #         i, j = self.fixed_pairs
        # x0 = x0[i]
        # x1 = x1[j]

        def cc_func(x0, x1, t):
            return self.cc(x0, x1, t, ids)
        vectors = velocity(cc_func, self.ts, x0, x1)
        cc_pts = self.cc(x0, x1, self.ts, ids)
        vectors_flat = vectors.flatten(0,1)
        cc_pts_flat = cc_pts.flatten(0, 1)
        jac_flat = jacobian(self.func, cc_pts_flat)

        len_loss = self.length_loss(vectors_flat, jac_flat)
        self.log(f'loss_length', len_loss, prog_bar=True, on_epoch=True)
        loss = self.length_weight * len_loss
        ts_expanded = self.ts.unsqueeze(1).unsqueeze(2)
        ts_expanded = ts_expanded.expand(-1, cc_pts.shape[1], -1)
        vt = self.flow_model(torch.cat([cc_pts, ts_expanded], dim=2).flatten(0,1))
        v_loss = ((vt - vectors_flat) ** 2).mean()
        self.log(f'fm_length', v_loss, prog_bar=True, on_epoch=True)

        loss = loss + self.flow_weight * v_loss
        if self.use_density and self.density_weight > 0 and self.data_pts is not None:
            density_loss = self.density_loss(cc_pts_flat, self.data_pts, cc_pts)
            self.log(f'density_loss', density_loss, prog_bar=True, on_epoch=True)
            loss += self.density_weight * density_loss
            
        self.log(f'loss', loss, prog_bar=True, on_epoch=True)
        
        return loss

    # def on_before_optimizer_step(self, optimizer):
    #     # check the gradient norm dl/dparams
    #     flow_model_norms = grad_norm(self.flow_model, norm_type=2)
    #     cc_norms = grad_norm(self.cc, norm_type=2)
    #     # import pdb; pdb.set_trace()
    #     # Plot the gradient norms as bar chart.

    #     flow_model_norms_arr = [norm.item() for (_, norm) in flow_model_norms.items()]
    #     cc_norms_arr = [norm.item() for (_, norm) in cc_norms.items()]
    #     fig = plt.figure()
    #     fig.add_subplot(1,2,1)
    #     plt.bar(range(len(flow_model_norms_arr)), flow_model_norms_arr)
    #     plt.title('Flow Model Grad Norms')
    #     fig.add_subplot(1,2,2)
    #     plt.bar(range(len(cc_norms_arr)), cc_norms_arr)
    #     plt.title('CondCurve Grad Norms')
    #     plt.savefig(f'./eb_fm/grad_norms/grad_norms_epoch_{self.current_epoch+1:04d}.png')

    
    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            print("Starting training at epoch 0")

            if self.visualize_training is False or self.dataloader is None or self.data_pts is None:
                return
    
            n_trajectories = 10  # Number of trajectories to sample and visualize

            # Generate trajectories
            ids = torch.zeros((self.visualize_x0.size(0), 1), device=self.visualize_x0.device, dtype=self.visualize_x0.dtype)
            with torch.no_grad():
                trajectories = self.cc(self.visualize_x0, self.visualize_x1, self.ts, ids) # (T, N, d)
                traj_z = self.encoder(trajectories.flatten(0,1))
                visualize_z0 = self.encoder(self.visualize_x0)
                visualize_z1 = self.encoder(self.visualize_x1)

            traj_z = traj_z.reshape(len(self.ts), self.visualize_x0.shape[0], -1).cpu().numpy()
            visualize_z0 = visualize_z0.cpu().numpy()
            visualize_z1 = visualize_z1.cpu().numpy()

            print('traj_z.shape: ', traj_z.shape)
            print('visualize_z0.shape: ', visualize_z0.shape)
            print('visualize_z1.shape: ', visualize_z1.shape)

            # Visualize trajectories
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=self.data_pts_encodings[:,0], y=self.data_pts_encodings[:,1], z=self.data_pts_encodings[:,2], 
                                    mode='markers', marker=dict(size=2, color='gray', colorscale='Viridis', opacity=0.8)))
            fig.add_trace(go.Scatter3d(x=visualize_z0[:,0], y=visualize_z0[:,1], z=visualize_z0[:,2],
                                        mode='markers', marker=dict(size=5, color='blue', opacity=0.8)))
            fig.add_trace(go.Scatter3d(x=visualize_z1[:,0], y=visualize_z1[:,1], z=visualize_z1[:,2],
                                        mode='markers', marker=dict(size=5, color='green', opacity=0.8)))
            
            for i in range(n_trajectories):
                fig.add_trace(go.Scatter3d(x=traj_z[:,i,0], y=traj_z[:,i,1], z=traj_z[:,i,2],
                                        mode='lines', line=dict(width=2, color='blue')))

            # Save the figure to a file
            os.makedirs(self.training_save_dir, exist_ok=True)
            fig.write_image(f'{self.training_save_dir}/trajs_epoch_{0:04d}.png')
    
    def on_train_epoch_end(self):
        if self.visualize_training is False or self.dataloader is None or self.data_pts is None:
            return
        
        n_trajectories = 10  # Number of trajectories to sample and visualize

        # Generate trajectories
        ids = torch.zeros((self.visualize_x0.size(0), 1), device=self.visualize_x0.device, dtype=self.visualize_x0.dtype)
        with torch.no_grad():
            trajectories = self.cc(self.visualize_x0, self.visualize_x1, self.ts, ids) # (T, N, d)
            traj_z = self.encoder(trajectories.flatten(0,1))
            visualize_z0 = self.encoder(self.visualize_x0)
            visualize_z1 = self.encoder(self.visualize_x1)

        traj_z = traj_z.reshape(len(self.ts), self.visualize_x0.shape[0], -1).cpu().numpy()
        visualize_z0 = visualize_z0.cpu().numpy()
        visualize_z1 = visualize_z1.cpu().numpy()

        # Visualize trajectories
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=self.data_pts_encodings[:,0], y=self.data_pts_encodings[:,1], z=self.data_pts_encodings[:,2], 
                                   mode='markers', marker=dict(size=2, color='gray', colorscale='Viridis', opacity=0.8)))
        fig.add_trace(go.Scatter3d(x=visualize_z0[:,0], y=visualize_z0[:,1], z=visualize_z0[:,2],
                                    mode='markers', marker=dict(size=5, color='blue', opacity=0.8)))
        fig.add_trace(go.Scatter3d(x=visualize_z1[:,0], y=visualize_z1[:,1], z=visualize_z1[:,2],
                                    mode='markers', marker=dict(size=5, color='green', opacity=0.8)))
        
        for i in range(n_trajectories):
            fig.add_trace(go.Scatter3d(x=traj_z[:,i,0], y=traj_z[:,i,1], z=traj_z[:,i,2],
                                       mode='lines', line=dict(width=2, color='blue')))

        # Save the figure to a file
        os.makedirs(self.training_save_dir, exist_ok=True)
        fig.write_image(f'{self.training_save_dir}/trajs_epoch_{self.current_epoch+1:04d}.png')
    
    # def on_train_end(self):
    #     if self.visualize_training is False:
    #         return
    #     self.frame_dir = self.training_save_dir
    #     self.video_output_path = f'{self.training_save_dir}/../trajs.mp4'
    #     # Create video from saved frames
    #     frames = [f for f in os.listdir(self.frame_dir) if f.endswith('.png')]
    #     frames.sort()
        
    #     clip = mpy.ImageSequenceClip([os.path.join(self.frame_dir, f) for f in frames], fps=2)
    #     clip.write_videofile(self.video_output_path, audio=False)

    #     # Clean up frame directory
    #     for f in frames:
    #         os.remove(os.path.join(self.frame_dir, f))
    #     os.rmdir(self.frame_dir)
