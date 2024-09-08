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

# class PointCloudDataset(torch.utils.data.Dataset):
#     """
#     Point Cloud Dataset
#     """
#     def __init__(self, pointcloud, distances):
#         self.pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
#         self.distances = torch.tensor(distances, dtype=torch.float32)

#     def __len__(self):
#         return len(self.pointcloud)
    
#     def __getitem__(self, idx):
#         return self.pointcloud[idx], idx  # Return point cloud and its index

# def make_custom_collate_fn(dataset):
#     def custom_collate_fn(batch):
#         pointclouds, indices = zip(*batch)
#         pointclouds = torch.stack(pointclouds)
        
#         # Access `distances` from the dataset
#         dist_mat = dataset.distances[torch.tensor(indices)][:, torch.tensor(indices)]
#         upper_tri_indices = np.triu_indices(dist_mat.size(0), k=1)
#         dist_mat_upper_tri = dist_mat[upper_tri_indices]
        
#         batch = {'x': pointclouds, 'd': dist_mat_upper_tri}
#         return batch
    
#     return custom_collate_fn


# TODO: make this more standard? Can (or should) I do batching in the dataloader instead?
class PointCloudDataset(torch.utils.data.Dataset):
    """
    Point Cloud Dataset
    """
    def __init__(self, pointcloud, distances, mask_d=None, mask_x=None, batch_size = 64, shuffle=True, pc_recon_weights=None):
        self.pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
        self.distances = torch.tensor(distances, dtype=torch.float32)
        self.mask_d = torch.tensor(mask_d, dtype=torch.float32) if mask_d is not None else None
        self.mask_x = torch.tensor(mask_x, dtype=torch.float32) if mask_x is not None else None
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.pc_recon_weights = pc_recon_weights

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
        if self.mask_d is not None:
            mask_d_mat = self.mask_d[batch_idxs][:,batch_idxs]
            batch['md'] = mask_d_mat[np.triu_indices(mask_d_mat.size(0), k=1)]
        if self.mask_x is not None:
            batch['mx'] = self.mask_x[batch_idxs]
        if self.pc_recon_weights is not None:
            batch['mw'] = self.pc_recon_weights
        return batch

def dataloader_from_pc(pointcloud, distances, batch_size = 64, shuffle=True, mask_d=None, mask_x=None, pc_recon_weights=None):
    dataset = PointCloudDataset(pointcloud=pointcloud, distances=distances, mask_d=mask_d, mask_x=mask_x, batch_size=batch_size, shuffle=shuffle, pc_recon_weights=pc_recon_weights)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=shuffle)

    return dataloader


# def dataloader_from_pc(pointcloud, distances, batch_size = 64, shuffle=True):
#     dataset = PointCloudDataset(pointcloud, distances)
#     custom_collate = make_custom_collate_fn(dataset)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate)
#     return dataloader

def train_valid_loader_from_pc(pointcloud, distances, 
                               batch_size = 64, train_valid_split = 0.8, 
                               shuffle=True, seed=42, return_mean_std=False, componentwise_std=False,
                               mask_d=None, mask_x=None, pc_recon_weights=None):
    X = pointcloud
    D = distances

    np.random.seed(seed)

    if shuffle:
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        D = D[idxs][:,idxs]
        if mask_d is not None:
            mask_d = mask_d[idxs][:,idxs]
        if mask_x is not None:
            mask_x = mask_x[idxs]
    split_idx = int(len(X)*train_valid_split)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    D_train = D[:split_idx,:split_idx]
    D_test = D[split_idx:,split_idx:]
    if mask_d is not None:
        mask_d_train = mask_d[:split_idx,:split_idx]
        mask_d_test = mask_d[split_idx:,split_idx:]
    else:
        mask_d_train = None
        mask_d_test = None
    if mask_x is not None:
        mask_x_train = mask_x[:split_idx]
        mask_x_test = mask_x[split_idx:]
    else:
        mask_x_train = None
        mask_x_test = None

    trainloader = dataloader_from_pc(X_train, D_train, batch_size, mask_d=mask_d_train, mask_x=mask_x_train, pc_recon_weights=pc_recon_weights)
    testloader = dataloader_from_pc(X_test, D_test, batch_size, mask_d=mask_d_test, mask_x=mask_x_test, pc_recon_weights=pc_recon_weights)

    if return_mean_std:
        if componentwise_std:
            std = X_train.std(axis=0)
        else:
            std = X_train.std()
        return trainloader, testloader, X_train.mean(axis=0), std
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
                 dist_normalization: str = "zscore",
                 emb_dim: int = 2,
                 knn: int = 5,
                 n_landmark: int = 5000,
                 t: int = 0, # 0 means auto;
                 shuffle: bool = True) -> None:
        super().__init__()
        self.data_name = data_name
        self.X = X.astype(np.float32)
        self.X_labels = X_labels
        self.pointcloud = torch.tensor(self.X) # for autometric interop
        self.dist_type = dist_type
        self.dist_normalization = dist_normalization
        self.emb_dim = emb_dim
        self.knn = knn
        self.n_landmark = n_landmark
        self.shuffle = shuffle

        if row_stochastic_matrix is not None:
            self.row_stochastic_matrix = row_stochastic_matrix
            self.phate_embed = phate_embed
        else:
            self.row_stochastic_matrix, self.phate_embed, self.phate_op, self.diff_pot = self._set_row_stochastic_matrix(t)
        
        if dist_type == 'phate_dist': # distance matching
            self.gt_dist = self._set_gt_dist()
        
        if t == 0:
            self.t = self.phate_op._find_optimal_t(t_max=100)
        else:
            self.t = t
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, index) -> Tuple[int, np.ndarray]:
        '''
            Returns index and the corresponding points
        '''
        return index, self.X[index]
    
    def __repr__(self) -> str:
        return f"RowStochasticDataset({self.data_name}, {self.X.shape}, {self.dist_type})"
    
    def _set_row_stochastic_matrix(self, t) -> Tuple[np.ndarray, np.ndarray]:
        '''
            Returns row_stochastic_matrix and phate_embed
            diff_potential = -1 * np.log(diff_op_t)
            diff_op_t = np.exp(-1 * diff_potential)
        '''
        if self.dist_type in ["phate_prob", "phate_dist"]:
            phate_op = phate.PHATE( 
                        verbose=True,
                        n_components=self.emb_dim,
                        knn=self.knn,
                        t=t if t > 0 else 'auto',
                        n_landmark=self.n_landmark).fit(self.X)
            phate_embed = torch.Tensor(phate_op.transform(self.X))
            diff_potential = phate_op.diff_potential
            diff_op_t = np.exp(-1 * diff_potential)
            row_stochastic_matrix = torch.Tensor(diff_op_t)
            print('row_stochastic_matrix', row_stochastic_matrix.shape)

            print('checking row sum:', np.allclose(row_stochastic_matrix.sum(axis=1), 1))
            print('row sum: ', row_stochastic_matrix.sum(axis=1)[:20])

            return row_stochastic_matrix, phate_embed, phate_op, diff_potential
        else:
            raise ValueError(f"dist_type {self.dist_type} not supported")
    
    def _set_gt_dist(self) -> np.ndarray:
        diff_pot = self.diff_pot

        # if self.dist_type == "phate":
        #     gt_dist = scipy.spatial.distance.cdist(diff_pot, diff_pot)
        #     if self.dist_normalization == "minmax":
        #         gt_dist = (gt_dist - gt_dist.min()) / (gt_dist.max() - gt_dist.min())
        #     elif self.dist_normalization == "zscore":
        #         gt_dist = (gt_dist - gt_dist.mean()) / gt_dist.std()
        #         # move to >= 0
        #         if gt_dist.min() < 0:
        #             gt_dist = gt_dist - gt_dist.min()

        # elif self.dist_type == "diffusion_potential":

        if self.dist_normalization == "minmax":
            diff_pot = (diff_pot - diff_pot.min()) / (diff_pot.max() - diff_pot.min())
        elif self.dist_normalization == "zscore":
            diff_pot = (diff_pot - diff_pot.mean()) / diff_pot.std()
            # move to >= 0
            if diff_pot.min() < 0:
                diff_pot = diff_pot - diff_pot.min()
                
        gt_dist = scipy.spatial.distance.cdist(diff_pot, diff_pot)

        return gt_dist

    def get_gt_dist(self, batch_idx) -> np.ndarray:
        '''
            Returns ground truth distance matrix
        '''
        if self.dist_type == "phate_prob":
            raise ValueError("for phate_prob, use row stochastic matrix instead of gt_dist!")
        
        return self.gt_dist[batch_idx][:, batch_idx] # [B, B]
    
        
