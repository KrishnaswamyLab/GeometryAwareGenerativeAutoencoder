'''
    Script for preparing data for the model
    Data: dict
    - data (X)
    - colors (label)
    - dist 
    - is_train (train_mask) 1 if train, 0 if test
    - data_gt (gt_X)
    - metadata (e.g. parameters for data generation)
'''
import os
import argparse
from typing import Literal, Optional
from glob import glob
from typing import List, Tuple
import hydra
from omegaconf import DictConfig, OmegaConf

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

def myeloid_data(fpath: str = '../raw_data/BMMC_myeloid.csv.gz',
                 save_path: str = '../raw_data/BMMC_myeloid.h5ad'):
    '''BMMC myeloid dataset'''
    if os.path.exists(save_path):
        adata = ad.read_h5ad(save_path)
        return None, adata.X, None
    
    myeloid_data = pd.read_csv(fpath, index_col=0, compression='gzip')
    adata = ad.AnnData(myeloid_data, 
                       obs=pd.DataFrame(index=myeloid_data.index), 
                       var=pd.DataFrame(index=myeloid_data.columns))
    scanpy.pp.recipe_seurat(adata)
    adata.write(save_path)

    return None, adata.X, None

def eb_data(fpath: str = '/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/data/eb_all.npz'):
    '''EB data'''
    data = np.load(fpath)
    X = data['data']
    labels = data['colors']

    return None, X, labels

def sea_ad_data(fpath: str = '/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/data/sea_ad.npz'):
    '''Sea-Ad data'''
    data = np.load(fpath)
    X = data['data']
    labels = data['colors']

    return None, X, labels


def tree_data(
        n_dim: int = 5,
        n_points: int = 500,
        n_branch: int = 5,
        noise: float = 1.0,
        random_state=2024):
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
    noise = np.random.normal(0, noise, gt_X.shape)
    X = gt_X + noise

    labels = np.array([i // n_points for i in range(n_branch * n_points)])
    
    return gt_X, X, labels

@hydra.main(version_base=None, config_path='../conf/data', config_name='sweep_data.yaml')
def gen_data(cfg: DictConfig) -> None:
    print(cfg)

    data = {
        'data_gt': None, # noiseless data
        'data': None,
        'colors': None, # label
    }
    metadata = {
        'n': cfg.n,
        'noise': cfg.noise,
    }

    seed = cfg.seed
    seed_everything(seed)
    
    if cfg.name == 'swiss_roll':
        # Generate Swiss Roll data
        gt_X, X, label = sklearn_swiss_roll(n_samples=cfg.n, noise=cfg.noise, random_state=cfg.seed)
    elif cfg.name == 's_curve':
        # Generate S-curve data
        gt_X, X, label = sklearn_s_curve(n_samples=cfg.n, noise=cfg.noise, random_state=cfg.seed)
    elif cfg.name == 'tree':
        # Generate tree data
        gt_X, X, label = tree_data(n_dim=5, n_points=500, n_branch=5, noise=cfg.noise, random_state=cfg.seed)
        cfg.n = 5 * 500 # Ugly hardcode, just to compare to HeatGeo
        metadata['n'] = 5 * 500
        metadata['n_dim'] = 5
        metadata['n_points'] = 500
        metadata['n_branch'] = 5
    elif cfg.name == 'myeloid':
        # Load BMMC myeloid data
        gt_X, X, label = myeloid_data()
    elif cfg.name == 'eb':
        # Load EB data
        gt_X, X, label = eb_data()
    elif cfg.name == 'sea_ad':
        # Load Sea-Ad data
        gt_X, X, label = sea_ad_data()
    else:
        raise ValueError(f'Unknown data: {cfg.name}')
    
    # Save the data
    data['data_gt'] = gt_X
    data['data'] = X
    data['colors'] = label

    # Generate is_train mask
    idxs = np.random.permutation(X.shape[0])
    split_idx = int(X.shape[0] * cfg.train_ratio)
    is_train = np.zeros(X.shape[0], dtype=int)
    is_train[idxs[:split_idx]] = 1
    data['is_train'] = is_train

    root_dir = '../data'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    saved_name = f'{cfg.name}.npz' if cfg.name in ['myeloid', 'eb', 'sea_ad'] else f'{cfg.name}_noise{cfg.noise}_seed{cfg.seed}.npz'
    np.savez(os.path.join(root_dir, saved_name), **data)

    check_data = np.load(os.path.join(root_dir, saved_name), allow_pickle=True)
    print(check_data['data_gt'].shape, 
          check_data['data'].shape, 
          check_data['colors'].shape,
          check_data['is_train'].shape)


if __name__ == '__main__':
    gen_data()

    # parser = argparse.ArgumentParser(description='Data preparation script')
    # parser.add_argument('--data', type=str, default='swiss_roll', help='Data to prepare')
    # parser.add_argument('--n', type=int, default=3000, help='Number of samples')
    # parser.add_argument('--noise', type=float, default=1.0, help='Noise level')
    # parser.add_argument('--train-ratio', type=float, default=0.8, help='Train-test split')
    # args = parser.parse_args()

    # data = {
    #     'data_gt': None, # noiseless data
    #     'data': None,
    #     'colors': None, # label
    # }
    # metadata = {
    #     'n': cfg.n,
    #     'noise': cfg.noise,
    # }
    
    # if cfg.name == 'swiss_roll':
    #     # Generate Swiss Roll data
    #     gt_X, X, label = sklearn_swiss_roll(n_samples=cfg.n, noise=cfg.noise, random_state=2024)
    # elif cfg.name == 's_curve':
    #     # Generate S-curve data
    #     gt_X, X, label = sklearn_s_curve(n_samples=cfg.n, noise=cfg.noise, random_state=2024)
    # elif cfg.name == 'tree':
    #     # Generate tree data
    #     gt_X, X, label = tree_data(n_dim=5, n_points=500, n_branch=5, noise=cfg.noise, random_state=2024)
    #     cfg.n = 5 * 500 # Ugly hardcode, just to compare to HeatGeo
    #     metadata['n'] = 5 * 500
    #     metadata['n_dim'] = 5
    #     metadata['n_points'] = 500
    #     metadata['n_branch'] = 5
    # elif cfg.name == 'myeloid':
    #     # Load BMMC myeloid data
    #     gt_X, X, label = myeloid_data()
    # else:
    #     raise ValueError(f'Unknown data: {cfg.name}')
    
    # # Save the data
    # data['data_gt'] = gt_X
    # data['data'] = X
    # data['colors'] = label

    # # generate is_train mask
    # idxs = np.random.permutation(X.shape[0])
    # split_idx = int(X.shape[0] * cfg.train_ratio)
    # is_train = np.zeros(X.shape[0], dtype=int)
    # is_train[idxs[:split_idx]] = 1
    # data['is_train'] = is_train

    # np.savez(f'../data/{cfg.name}_noise{cfg.noise}.npz', **data)

    
    # check_data = np.load(f'../data/{cfg.name}_noise{cfg.noise}.npz', allow_pickle=True)
    # print(check_data['data_gt'].shape, 
    #       check_data['data'].shape, 
    #       check_data['colors'].shape,
    #       check_data['is_train'].shape)
