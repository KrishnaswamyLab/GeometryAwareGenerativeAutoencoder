'''
    Compare geodesic distances of different models.

'''

import os
import sys
import numpy as np
import pandas as pd
import torch
import phate
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
import networkx as nx

sys.path.append('../../src/')
from diffusionmap import DiffusionMap

def _diffusion_map_dist(X, start_points, end_points, k=5, latent_dim=3, seed=42):
    '''

    Returns:
        distances: list of geodesic distances for each pair of start and end points.
    '''
    dm = DiffusionMap(n_components=latent_dim, seed=seed)
    embeddings = dm.fit_transform(X)
    print('embeddings: ', embeddings.shape)

    # Compute geoedisc distances between each pair of points
    distances = []
    for start, end in zip(start_points, end_points):
        start_idx = int(np.argmin(np.linalg.norm(X - start, axis=1)))
        end_idx = int(np.argmin(np.linalg.norm(X - end, axis=1)))

        dist = np.linalg.norm(embeddings[start_idx] - embeddings[end_idx])
        distances.append(dist)

    return distances

def _phate_dist(X, start_points, end_points, k=5, latent_dim=3, seed=42):
    '''

    Returns:
        distances: np.ndarray of shape [N, N] pairwise geodesic distances.
    '''
    phate_op = phate.PHATE(n_components=latent_dim, random_state=seed, knn=k)
    embeddings = phate_op.fit_transform(X)
    print('embeddings: ', embeddings.shape) 

    # Compute geoedisc distances between each pair of points
    distances = []
    for start, end in zip(start_points, end_points):
        start_idx = int(np.argmin(np.linalg.norm(X - start, axis=1)))
        end_idx = int(np.argmin(np.linalg.norm(X - end, axis=1)))

        dist = np.linalg.norm(embeddings[start_idx] - embeddings[end_idx])
        distances.append(dist)

    return distances

def _djikstra_dist(X, start_points, end_points, k=5, latent_dim=3, seed=42):
    '''
        Compute geodesic distances using Djikstra algorithm.

    Returns:
        lengths: list of geodesic distances for each pair of start and end points.
    '''
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(start_points, np.ndarray):
        start_points = torch.from_numpy(start_points).float()
    if isinstance(end_points, np.ndarray):
        end_points = torch.from_numpy(end_points).float()

    # get start and end points indices
    a_indices = []
    b_indices = []
    for i in range(start_points.shape[0]):
        a_idx = int(torch.argmin(torch.linalg.norm(X - start_points[i], dim=1), dim=0))
        b_idx = int(torch.argmin(torch.linalg.norm(X - end_points[i], dim=1), dim=0))
        a_indices.append(a_idx)
        b_indices.append(b_idx)
    
    assert len(a_indices) == len(b_indices) == start_points.shape[0]

    # get k-nn
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
        
    G = nx.Graph()
    # add nodes to G
    for i in range(len(X)):
        G.add_node(i)
    # Add edges between each point and its k-nearest neighbors
    for i in range(len(X)):
        for j in range(1, k+1):  # start from 1 to avoid self-loop (i.e., point itself)
            G.add_edge(i, indices[i][j], weight = torch.linalg.norm(X[i] - X[indices[i][j]]))
            G.add_edge(indices[i][j], i, weight = torch.linalg.norm(X[i] - X[indices[i][j]]))
    
    
    # find the shortest path between a and b
    lengths = []
    for i in range(start_points.shape[0]):
        a_idx = a_indices[i]
        b_idx = b_indices[i]

        #path = nx.shortest_path(G, a_idx, b_idx, weight = "weight")
        length = nx.shortest_path_length(G, a_idx, b_idx, weight = "weight")
        lengths.append(length)
    
    #g = X_combined[path]
    return lengths


if __name__ == '__main__':
    data_root = '../experiment_data' # TODO: change to your correct path
    data_paths = os.listdir(data_root)
    data_paths.sort()
    dataset_names = [file.split('.npz')[0] for file in data_paths]
    #dataset_name = 'hemisphere_none_0.1'
    methods = 'diffusion_map,djikstra'
    methods = methods.split(',')
    k = 5
    latent_dim = 3
    seed = 42

    for dataset_name in dataset_names:
        print(f"Dataset: {dataset_name}")

        # Load data.
        data_files = np.load(f"{data_root}/{dataset_name}.npz", allow_pickle=True)
        print(f"Loaded data files: {data_files.files}")

        X = data_files['X']
        start_points = data_files['start_points']
        end_points = data_files['end_points']

        # Compute geodesic distances.
        pred_lengths = []
        for method in methods:
            mfunc = globals()[f"_{method}_dist"]
            distances = mfunc(X, start_points, end_points, k=5, latent_dim=latent_dim, seed=seed)
            pred_lengths.append(distances)
            print(f"Method: {method}, Geodesic distances: {len(distances)}")
        
        # Compare with ground-truth.
        gt_geodesic = data_files['geodesic_lengths']
        print(f"Ground-truth geodesic distances: {gt_geodesic.shape}")

        # correlation between predicted and ground-truth geodesic distances
        df = pd.DataFrame() # columns: method, pearson_corr, pearson_p, spearman_corr, spearman_p
        for method in methods:
            pred_geodesic = np.array(pred_lengths[methods.index(method)])
            pr, pp = pearsonr(pred_geodesic, gt_geodesic)
            sr, sp = spearmanr(pred_geodesic, gt_geodesic)
            df = df.append({'dataset_name': dataset_name, 'method': method, 
                            'pearson_corr': pr, 'pearson_p': pp, 'spearman_corr': sr, 'spearman_p': sp}, ignore_index=True)

            print(f"{method}: Pearson corr: {pr} with prob {pp}, Spearman corr {sr} with {pp}")

        # Save to df
        save_path = "./geodesic_comparison.csv"
        if os.path.exists(save_path):
            os.remove(save_path)
        
        df.to_csv(save_path, index=False)

    print('Done.')

