"""
Adapted From https://github.com/professorwug/autometric
"""
from typing import Optional, Tuple

import torch
import numpy as np
import phate
import scipy
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

# TODO: make this more standard? Can (or should) I do batching in the dataloader instead?
class PointCloudDataset(torch.utils.data.Dataset):
    """
    Point Cloud Dataset
    """
    def __init__(self, pointcloud, distances, batch_size = 64, shuffle=True):
        self.pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
        self.distances = torch.tensor(distances, dtype=torch.float32)
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __len__(self):
        return len(self.pointcloud)
    
    def __getitem__(self, idx):
        '''
            Returns a batch of pointclouds and their distances
            batch['x'] = [B, D]
            batch['d'] = [B, B(B-1)/2] (upper triangular), assuming symmetric distance matrix
        '''
        if self.shuffle:
            # TODO: [fix] generate random permutation once and use it for all batches
            batch_idxs = torch.randperm(len(self.pointcloud))[:self.batch_size] 
        else:
            batch_idxs = torch.arange(idx, idx+self.batch_size) % len(self.pointcloud)
        batch = {}
        batch['x'] = self.pointcloud[batch_idxs]
        dist_mat = self.distances[batch_idxs][:,batch_idxs]
        batch['d'] = dist_mat[np.triu_indices(dist_mat.size(0), k=1)]

        return batch

def dataloader_from_pc(pointcloud, distances, batch_size = 64, shuffle=True):
    dataset = PointCloudDataset(pointcloud, distances, batch_size, shuffle=shuffle)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=shuffle)

    return dataloader

def train_valid_loader_from_pc(pointcloud, distances, 
                               batch_size = 64, train_valid_split = 0.8, 
                               shuffle=True, seed=42, return_mean_std=False):
    X = pointcloud
    D = distances

    np.random.seed(seed)

    if shuffle:
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        D = D[idxs][:,idxs]
    split_idx = int(len(X)*train_valid_split)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    D_train = D[:split_idx,:split_idx]
    D_test = D[split_idx:,split_idx:]
    
    trainloader = dataloader_from_pc(X_train, D_train, batch_size)
    testloader = dataloader_from_pc(X_test, D_test, batch_size)

    if return_mean_std:
        return trainloader, testloader, X_train.mean(axis=0), X_train.std(axis=0)
    return trainloader, testloader

def train_valid_test_loader_from_pc(
    pointcloud, distances, 
    batch_size = 64, train_test_split = 0.8, train_valid_split = 0.8, 
    shuffle=True, seed=42
):
    X = pointcloud
    D = distances

    np.random.seed(seed)

    if shuffle:
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        D = D[idxs][:,idxs]
    split_idx = int(len(X)*train_test_split)
    split_val_idx = int(split_idx*train_valid_split)
    X_train = X[:split_val_idx]
    X_valid = X[split_val_idx:split_idx]
    X_test = X[split_idx:]
    D_train = D[:split_val_idx,:split_val_idx]
    D_valid = D[split_val_idx:split_idx,split_val_idx:split_idx]
    D_test = D[split_idx:,split_idx:]

    trainloader = dataloader_from_pc(X_train, D_train, batch_size)
    validloader = dataloader_from_pc(X_valid, D_valid, batch_size)
    testloader = dataloader_from_pc(X_test, D_test, batch_size)

    return trainloader, validloader, testloader

class RowStochasticDataset(torch.utils.data.Dataset):
    '''
        Dataset for point cloud and its row stochastic transition matrix
        dist_type: "phate_prob"

        example:
            X, thetas = swiss_roll_data()
            dataset = RowStochasticDataset(data_name='swiss_roll', X=X, dist_type='phate')
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for batch_idx, (idx, data) in enumerate(data_loader):
                ...
                gt_row_stochastic_matrix = dataset.row_stochastic_matrix
                ...
    '''
    def __init__(self,
                 data_name: str,
                 X: np.ndarray,
                 X_labels: Optional[np.ndarray] = None,
                 row_stochastic_matrix: Optional[np.ndarray] = None,
                 phate_embed: Optional[np.ndarray] = None,
                 dist_type: str = "phate_prob",
                 emb_dim: int = 2,
                 knn: int = 5,
                 shuffle: bool = True) -> None:
        super().__init__()
        self.data_name = data_name
        self.X = X.astype(np.float32)
        self.X_labels = X_labels
        self.dist_type = dist_type
        self.emb_dim = emb_dim
        self.knn = knn
        self.shuffle = shuffle

        if row_stochastic_matrix is not None:
            self.row_stochastic_matrix = row_stochastic_matrix
            self.phate_embed = phate_embed
        else:
            self.row_stochastic_matrix, self.phate_embed, self.phate_op = self._set_row_stochastic_matrix()
        
        self.t = self.phate_op._find_optimal_t(t_max=100)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, index) -> Tuple[int, np.ndarray]:
        '''
            Returns index and the corresponding points
        '''
        return index, self.X[index]
    
    def __repr__(self) -> str:
        return f"RowStochasticDataset({self.data_name}, {self.X.shape}, {self.dist_type})"
    
    def _set_row_stochastic_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
            Returns row_stochastic_matrix and phate_embed
            diff_potential = -1 * np.log(diff_op_t)
            diff_op_t = np.exp(-1 * diff_potential)
        '''
        if self.dist_type == "phate_prob":
            phate_op = phate.PHATE(random_state=1, 
                        verbose=True,
                        n_components=self.emb_dim,
                        knn=self.knn,
                        n_landmark=self.X.shape[0]).fit(self.X)
            phate_embed = torch.Tensor(phate_op.transform(self.X))
            diff_potential = phate_op.diff_potential
            diff_op_t = np.exp(-1 * diff_potential)
            row_stochastic_matrix = torch.Tensor(diff_op_t)
            print('row_stochastic_matrix', row_stochastic_matrix.shape)

            print('checking row sum:', np.allclose(row_stochastic_matrix.sum(axis=1), 1))
            print('row sum: ', row_stochastic_matrix.sum(axis=1)[:20])

            return row_stochastic_matrix, phate_embed, phate_op
        else:
            raise ValueError(f"dist_type {self.dist_type} not supported")
        
