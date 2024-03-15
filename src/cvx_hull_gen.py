import numpy as np
import scprep
import torch
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_add

class CvxHullGenDirichlet():
    def __init__(self, k=10, alpha=1., sigma=0., p=.5):
        self.k = k
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        self.is_fit_ = False
        
    def fit(self, points):
        self.points = points
        self.device = points.device
        points = points.cpu()
        self.edge_index = knn_graph(points, self.k, loop=False).to(self.device)
        self.is_fit_ = True

    def generate(self, n_samples):
        assert self.is_fit_
        # randomly selecting reference neighbours
        weights = torch.ones(self.points.size(0), dtype=torch.float, device=self.device)
        indices = torch.multinomial(weights, n_samples, replacement=True)
        # generate a Dirichlet distribution of coefficients
        alpha_tensor = (torch.ones(self.k)*self.alpha).to(self.device)
        coefficients = torch.distributions.Dirichlet(alpha_tensor).sample((n_samples,))
        # add Gaussian noise to the coefficients

        filtered_edge_index, filtered_points = self.filter_edge_index_points(indices, self.edge_index, self.points)
        bernoulli_dist = torch.distributions.Bernoulli(self.p)
        berns = bernoulli_dist.sample(coefficients.size()).to(self.device)
        berns[berns.sum(axis=1)==0, 0] = 1
        coefficients = coefficients * berns
        coefficients = coefficients / coefficients.sum(axis=1, keepdim=True)
        noise = torch.randn_like(coefficients) * self.sigma
        coefficients += noise
        coef_flat = coefficients.view(-1)        
        generated = scatter_add((self.points[filtered_edge_index[0],:].T * coef_flat), filtered_edge_index[1]).T
        return generated

    def filter_edge_index_points(self, indices, edge_index, points):
        filtered_edge_index = []
        for i in range(len(indices)):
            sub_edge_index = edge_index[:,edge_index[1,:] == indices[i]]
            sub_edge_index[1,:] = i
            filtered_edge_index.append(sub_edge_index)
        filtered_edge_index = torch.cat(filtered_edge_index, dim=1)
        filtered_points = points[indices]
        return filtered_edge_index, filtered_points

class CvxHullGen():
    def __init__(self, k=10, alpha=1., sigma=0., p=0.5, delta=0.2):
        self.k = k
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        assert delta <= 1 and delta >= 0
        self.delta = delta
        self.is_fit_ = False
        
    def fit(self, points):
        self.points = points
        self.device = points.device
        points = points.cpu()
        self.edge_index = knn_graph(points, self.k, loop=False).to(self.device)
        self.is_fit_ = True

    def generate(self, n_samples):
        assert self.is_fit_
        # randomly selecting reference neighbours
        weights = torch.ones(self.points.size(0), dtype=torch.float, device=self.device)
        indices = torch.multinomial(weights, n_samples, replacement=True)
        coefficients = torch.rand(n_samples, self.k, device=self.device)
        coefficients = coefficients * (1 + 2 * self.delta) - self.delta
        coefficients[:,-1] = coefficients[:,:-1].sum(axis=1)
        filtered_edge_index, filtered_points = self.filter_edge_index_points(indices, self.edge_index, self.points)
        bernoulli_dist = torch.distributions.Bernoulli(self.p)
        berns = bernoulli_dist.sample(coefficients.size()).to(self.device)
        berns[berns.sum(axis=1)==0, 0] = 1
        coefficients = coefficients * berns
        coefficients = coefficients / coefficients.sum(axis=1, keepdim=True)
        noise = torch.randn_like(coefficients) * self.sigma
        coefficients *= 1+noise
        coef_flat = coefficients.view(-1)        
        generated = scatter_add((self.points[filtered_edge_index[0],:].T * coef_flat), filtered_edge_index[1]).T
        return generated

    def filter_edge_index_points(self, indices, edge_index, points):
        filtered_edge_index = []
        for i in range(len(indices)):
            sub_edge_index = edge_index[:,edge_index[1,:] == indices[i]]
            sub_edge_index[1,:] = i
            filtered_edge_index.append(sub_edge_index)
        filtered_edge_index = torch.cat(filtered_edge_index, dim=1)
        filtered_points = points[indices]
        return filtered_edge_index, filtered_points

