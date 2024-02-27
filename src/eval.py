'''
    Unified evaluation script for all models
    Data: dict
    - data
    - colors
    - dist
    - is_train
    - data_gt
    - metadata (e.g. parameters for data generation)
'''
import os
import numpy as np
import torch
import torch.nn as nn
import graphtools

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

synthetic_path = '../data/synthetic'
biodata_path = '../data/biodata'

def DEMaP(data, embedding, knn=10, subsample_idx=None) -> float:
    geodesic_dist = geodesic_distance(data, knn=knn)
    if subsample_idx is not None:
        geodesic_dist = geodesic_dist[subsample_idx, :][:, subsample_idx]
    geodesic_dist = squareform(geodesic_dist)
    embedded_dist = pdist(embedding)

    return spearmanr(geodesic_dist, embedded_dist).correlation

def geodesic_distance(data, knn=10, distance="data") -> np.ndarray:
    G = graphtools.Graph(data, knn=knn, decay=None)

    return G.shortest_path(distance=distance)

def _evaluate(model: nn.Module, data: dict, knn: int, subsample_idx=None) -> float:
    data_gt = data['data_gt']
    if data_gt is None:
        raise ValueError("Ground truth data not found in data dictionary")
    
    assert hasattr(model, 'encode'), "Model must have encode method"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_data = data['data']
    loader = torch.utils.data.DataLoader(input_data, batch_size=32, shuffle=False)

    model.eval()
    X = []
    Z = []
    for _, x in enumerate(loader):
        x = x.to(device)
        z = model.encode(x).detach().cpu().numpy()
        X.append(x)
        Z.append(z)
    X = np.concatenate(X, axis=0)
    Z = np.concatenate(Z, axis=0)
    assert X.shape[0] == Z.shape[0], "Input and latent space must have same number of samples"

    # Calculate DEMaP score
    demap = DEMaP(data_gt, Z, knn=knn, subsample_idx=None)
    
    return demap

def evaluate(model: nn.Module, cfg: dict):
    # for each data in folder
    # load data
    # demap = _evaluate(model, data, cfg.knn)
    pass

if __name__ == '__main__':
    pass

    