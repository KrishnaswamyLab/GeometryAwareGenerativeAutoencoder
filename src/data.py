"""
Adapted From https://github.com/professorwug/autometric
"""
import torch
import numpy as np

class PointCloudDataset(torch.utils.data.Dataset):
    """
    Point Cloud Dataset
    """
    def __init__(self, pointcloud, distances, batch_size = 64):
        self.pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
        self.distances = torch.tensor(distances, dtype=torch.float32)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.pointcloud)
    
    def __getitem__(self, idx):
        batch_idxs = torch.randperm(len(self.pointcloud))[:self.batch_size]
        batch = {}
        batch['x'] = self.pointcloud[batch_idxs]
        dist_mat = self.distances[batch_idxs][:,batch_idxs]
        batch['d'] = dist_mat[np.triu_indices(dist_mat.size(0), k=1)]    
        return batch

def dataloader_from_pc(pointcloud, distances, batch_size = 64):
    dataset = PointCloudDataset(pointcloud, distances, batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=True)
    return dataloader

def train_and_testloader_from_pc(
    pointcloud, distances, batch_size = 64, train_test_split = 0.8
):
    X = pointcloud
    D = distances
    split_idx = int(len(X)*train_test_split)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    D_train = D[:split_idx,:split_idx]
    D_test = D[split_idx:,split_idx:]
    trainloader = dataloader_from_pc(X_train, D_train, batch_size)
    testloader = dataloader_from_pc(X_test, D_test, batch_size)
    return trainloader, testloader