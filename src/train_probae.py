from typing import Tuple
import os
import matplotlib.pyplot as plt
import wandb
import hydra

import numpy as np
import pandas as pd
import torch
import scipy.sparse
from scipy.spatial.distance import pdist, squareform
from omegaconf import DictConfig, OmegaConf

from data import RowStochasticDataset
from model import AEProb
from metrics import distance_distortion, mAP
from procrustes import Procrustes
from utils.early_stop import EarlyStopping
from visualize import visualize

@hydra.main(version_base=None, config_path='../conf', config_name='config')
def train_eval(cfg: DictConfig):
    # if cfg.logger.use_wandb:
    #     config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    #     run = wandb.init(
    #         entity=cfg.logger.entity,
    #         project=cfg.logger.project,
    #         tags=cfg.logger.tags,
    #         reinit=True,
    #         config=config,
    #         settings=wandb.Settings(start_method="thread"),
    #     )

    print(cfg)

    ''' Data '''
    data_path = os.path.join(cfg.data.root, cfg.data.name + cfg.data.filetype)
    data = np.load(data_path, allow_pickle=True)
    
    raw_data = data['X']
    labels = data['label']

    assert raw_data.shape[0] == labels.shape[0]

    print(f'Done loading raw data. Raw data shape: {raw_data.shape}')

    shuffle = cfg.training.shuffle
    train_test_split = cfg.training.train_test_split
    train_valid_split = cfg.training.train_valid_split
    # if shuffle:
    #     idxs = np.random.permutation(len(raw_data))
    #     raw_data = raw_data[idxs]
    #     labels = labels[idxs]

    split_idx = int(len(raw_data)*train_test_split)
    split_val_idx = int(split_idx*train_valid_split)
    train_data = raw_data[:split_val_idx]
    train_val_data = raw_data[:split_idx]

    train_dataset = RowStochasticDataset(data_name=cfg.data.name, X=train_data, X_labels=None, dist_type='phate_prob')
    train_val_dataset = RowStochasticDataset(data_name=cfg.data.name, X=train_val_data, X_labels=None, dist_type='phate_prob')
    whole_dataset = RowStochasticDataset(data_name=cfg.data.name, X=raw_data, X_labels=None, dist_type='phate_prob')
    
    print(f'Train dataset: {len(train_dataset)}; \
          Val dataset: {len(train_val_dataset)}; \
          Whole dataset: {len(whole_dataset)}')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    whole_loader = torch.utils.data.DataLoader(whole_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    
    ''' Model '''
    emb_dim = 2
    activation_dict = {
        'relu': torch.nn.ReLU(),
        'leaky_relu': torch.nn.LeakyReLU(),
        'sigmoid': torch.nn.Sigmoid()
    }
    act_fn = activation_dict[cfg.model.activation]
    model = AEProb(dim=raw_data.shape[1], emb_dim=emb_dim, 
                     layer_widths=cfg.model.layer_widths, activation_fn=act_fn,
                     prob_method=cfg.model.prob_method, dist_reconstr_weights=cfg.model.dist_reconstr_weights)

    ''' Training '''
    device = cfg.training.accelerator
    epoch = cfg.training.max_epochs
    batch_size = cfg.training.batch_size
    lr = cfg.training.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopper = EarlyStopping(mode='min',
                                  patience=cfg.training.patience,
                                  percentage=False)

    best_metric = np.inf
    model = model.to(device)
    print('Training ...')
    for eid in range(epoch):
        model.train()
        epoch_loss = 0.0
        train_Z = []
        train_indices = []
        for (batch_inds, batch_x) in train_loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            batch_x_hat, batch_z = model(batch_x)
            train_Z.append(batch_z)
            train_indices.append(batch_inds.reshape(-1,1))

            # reconstruction loss per batch
            decoder_loss = model.decoder_loss(batch_x, batch_x_hat)

            epoch_loss += decoder_loss
        
        # row-wise prob divergence loss
        train_Z = torch.cat(train_Z, dim=0) #[N, emb_dim]
        #print('train_Z shape:', train_Z.shape)
        train_indices = torch.squeeze(torch.cat(train_indices, dim=0)) # [N,]
        pred_prob_matrix = model.compute_prob_matrix(train_Z)
        gt_prob_matrix = train_dataset.row_stochastic_matrix
        encoder_loss = model.encoder_loss(gt_prob_matrix, pred_prob_matrix)
        epoch_loss += encoder_loss

        epoch_loss /= len(train_loader)

        epoch_loss.backward()
        optimizer.step()

        print('\r[Epoch %d] Loss: %.4f' % (eid, epoch_loss.item()), end='')

        model.eval()
        val_loss = 0
        val_Z = []
        val_indices = []
        with torch.no_grad():
            for (batch_inds, batch_x) in train_val_loader:
                batch_x = batch_x.to(device)

                batch_x_hat, batch_z = model(batch_x)
                val_Z.append(batch_z)
                val_indices.append(batch_inds.reshape(-1,1)) # [B,1]
                val_loss += model.decoder_loss(batch_x, batch_x_hat).item()
            
            val_Z = torch.cat(val_Z, dim=0)
            val_indices = torch.squeeze(torch.cat(val_indices, dim=0)) # [N,]

            train_val_pred_prob_matrix = model.compute_prob_matrix(val_Z)
            gt_train_val_prob_matrix = train_val_dataset.row_stochastic_matrix
            val_encoder_loss = model.encoder_loss(gt_train_val_prob_matrix, 
                                                  train_val_pred_prob_matrix)
            val_loss += val_encoder_loss.item()
            
        if val_loss < best_metric:
            print('\nBetter model found. Saving best model ...\n')
            best_metric = val_loss
            best_model = model.state_dict()
            if cfg.path.save:
                torch.save(best_model, os.path.join(cfg.path.root, cfg.path.model))
            
        # Early Stopping
        if early_stopper.step(val_loss):
            print('Early stopping criterion met. Ending training.\n')
            break

    ''' Evaluation ''' 
    model.load_state_dict(best_model)

    distortion, disp, demap = evaluate(model,
                                       torch.from_numpy(raw_data).type(torch.float32),
                                       whole_dataset.phate_embed,
                                       whole_dataset.row_stochastic_matrix,
                                       dist_type='prob')
    print(f'Distortion: {distortion}, Disparity: {disp}, DeMAP: {demap}')

    ''' Visualize '''
    visualize(model=model,
              dataset_name=cfg.data.name,
              data=torch.from_numpy(raw_data).type(torch.float32),
              data_clusters=np.squeeze(labels),
              phate_embed=whole_dataset.phate_embed,
              gt_dist=whole_dataset.row_stochastic_matrix,
              dist_type='prob',
              metrics={'Distortion': distortion, 'Disparity': disp, 'DeMAP': demap},
              save_path=os.path.join(cfg.path.root, cfg.model.prob_method+'_embeddings.png'))



def evaluate(model: torch.nn.Module, 
             input_data: torch.Tensor,
             phate_embed: np.ndarray,
             gt_dist: torch.Tensor,
             dist_type: str = 'prob') -> Tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        pred_embed = model.encode(input_data)
        if dist_type == 'prob':
            pred_dist = model.compute_prob_matrix(pred_embed) # [N, N]
        else:
            pred_dist = torch.cdist(pred_embed, pred_embed) # [N, N]

    # distortion
    distortion = distance_distortion(pred_dist.cpu().detach().numpy(), 
                                     gt_dist.cpu().detach().numpy())
        
    # procrustes disparity
    procrustes_op = Procrustes()
    _, _, disparity = procrustes_op.fit_transform(phate_embed,
                                                pred_embed.cpu().detach().numpy())
    
    # TODO: DeMAP from phate paper

    return distortion, disparity, 0.0



if __name__ == "__main__":
    train_eval()