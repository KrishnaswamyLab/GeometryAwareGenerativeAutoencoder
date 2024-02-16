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
import anndata as ad
import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import sklearn.datasets

def sklearn_swiss_roll(n_samples, noise=0.0, random_state=0):
    X, t = sklearn.datasets.make_swiss_roll(n_samples, noise=noise, random_state=random_state)

    return X, t

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

    return adata.X


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preparation script')
    parser.add_argument('--data', type=str, default='swiss_roll', help='Data to prepare')
    parser.add_argument('--n', type=int, default=3000, help='Number of samples')
    args = parser.parse_args()

    data = {
        'X': None,
        'label': None,
    }
    
    if args.data == 'swiss_roll':
        # Generate Swiss Roll data
        X, t = sklearn_swiss_roll(n_samples=args.n, noise=0.1, random_state=0)
        data['X'] = X
        data['label'] = t
        np.savez('../data/swiss_roll.npz', **data)
    
    check_data = np.load('../data/swiss_roll.npz', allow_pickle=True)
    print(check_data['X'].shape, check_data['label'].shape)
