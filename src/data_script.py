'''
    Script for preparing data for the model
'''
import os
import argparse
from typing import Literal, Optional
from glob import glob
from typing import List, Tuple

import phate
import scipy
import scanpy
import pygsp
import graphtools
import anndata as ad
import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import sklearn.datasets
from scipy.sparse.csgraph import shortest_path

from utils.seed import seed_everything

seed = 2024
seed_everything(seed)

'''
    All data fuctions should return the following:
    gt_X: np.ndarray
    X: np.ndarray
    label: np.ndarray
'''

def sklearn_s_curve(n_samples, noise=0.0, random_state=2024):
    # Generate S-curve data without noise
    gt_X, t = sklearn.datasets.make_s_curve(n_samples, noise=0.0, random_state=random_state)

    # Add noise to the data
    noise = np.random.normal(0, noise, gt_X.shape)
    X = gt_X + noise

    return gt_X, X, t

def sklearn_swiss_roll(n_samples, noise=0.0, random_state=2024):
    # Generate Swiss Roll data without noise
    gt_X, t = sklearn.datasets.make_swiss_roll(n_samples, noise=0.0, random_state=random_state)

    # Add noise to the data
    noise = np.random.normal(0, noise, gt_X.shape)
    X = gt_X + noise

    return gt_X, X, t

def myeloid_data(fpath: str = '../raw_data/BMMC_myeloid.csv',
                 save_path: str = '../raw_data/BMMC_myeloid.h5ad'):
    '''BMMC myeloid dataset'''
    if os.path.exists(save_path):
        adata = ad.read_h5ad(save_path)
        return adata.X
    
    myeloid_data = pd.read_csv(fpath, index_col=0)
    adata = ad.AnnData(myeloid_data, 
                       obs=pd.DataFrame(index=myeloid_data.index), 
                       var=pd.DataFrame(index=myeloid_data.columns))
    scanpy.pp.recipe_seurat(adata)
    adata.write(save_path)

    return None, adata.X, None


def tree_data(
        n_dim: int = 10,
        n_points: int = 200,
        n_branch: int = 10,
        manifold_noise: float = 4,
        random_state=2024,
        clustered = None,
        train_fold = None):
    '''
    Generate tree data
    The geodeisc distances are computed from a manifold without noise 
    and the data are a noisy version.
    '''

    # The manifold witout noise
    gt_X, labels = phate.tree.gen_dla(
        n_dim=n_dim,
        n_branch=n_branch,
        branch_length=n_points,
        sigma=0,
        seed=random_state,
    )

    # The noisy manifold
    noise = np.random.normal(0, manifold_noise, gt_X.shape)
    X = gt_X + noise

    labels = np.array([i // n_points for i in range(n_branch * n_points)])
    
    return gt_X, X, labels


# def compute_geodesic_distances(data, knn=10, distance="data"):
#     G = graphtools.Graph(data, knn=knn, decay=None)
#     return G.shortest_path(distance=distance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preparation script')
    parser.add_argument('--data', type=str, default='swiss_roll', help='Data to prepare')
    parser.add_argument('--n', type=int, default=3000, help='Number of samples')
    args = parser.parse_args()

    data = {
        'gt_X': None, # noiseless data
        'X': None,
        'label': None,
    }
    
    if args.data == 'swiss_roll':
        # Generate Swiss Roll data
        gt_X, X, label = sklearn_swiss_roll(n_samples=args.n, noise=1.0, random_state=2024)
    elif args.data == 's_curve':
        # Generate S-curve data
        gt_X, X, label = sklearn_s_curve(n_samples=args.n, noise=1.0, random_state=2024)
    elif args.data == 'tree':
        # Generate tree data
        gt_X, X, label = tree_data(n_dim=10, n_points=400, n_branch=5, manifold_noise=1.0)
    elif args.data == 'myeloid':
        # Load BMMC myeloid data
        gt_X, X, label = myeloid_data()
    else:
        raise ValueError(f'Unknown data: {args.data}')
    
    # Save the data
    data['gt_X'] = gt_X
    data['X'] = X
    data['label'] = label
    np.savez(f'../data/{args.data}.npz', **data)

    
    check_data = np.load(f'../data/{args.data}.npz', allow_pickle=True)
    print(check_data['gt_X'].shape, check_data['X'].shape, check_data['label'].shape)
