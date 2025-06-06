from typing import Tuple
import os
import matplotlib.pyplot as plt
import wandb
import hydra
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import torch
import pygsp
import scipy.sparse
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from omegaconf import DictConfig, OmegaConf

from data import RowStochasticDataset
from model import AEProb
from metrics import distance_distortion, mAP, computeKNNmAP
from procrustes import Procrustes
from utils.early_stop import EarlyStopping
from utils.log_utils import log
from utils.seed import seed_everything
from visualize import visualize
# import demap

import graphtools
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

load_dotenv('../.env')
PROJECT_PATH=os.getenv('PROJECT_PATH')
WANDB_ENTITY=os.getenv('WANDB_ENTITY')

@hydra.main(version_base=None, config_path='../conf', config_name='probae_config')
def train_eval(cfg: DictConfig):
    run = None
    if cfg.logger.use_wandb:
        config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=cfg.logger.project,
            tags="probae",
            name=f'{cfg.model.prob_method}_{cfg.data.name}',
            reinit=True,
            config=config,
            settings=wandb.Settings(start_method="thread"),
        )
    log(cfg)

    # Seed everything
    seed_everything(cfg.training.seed)

    ''' Data '''
    true_data = None
    if cfg.data.name == 'demap':
        splatter_data = np.load('../data/splatter.npz', allow_pickle=True)
        true_data = splatter_data['true']
        noisy_data = splatter_data['noisy']
        raw_data = noisy_data
        labels = None
    else:
        data_path = os.path.join(PROJECT_PATH, cfg.data.root, f'{cfg.data.name}_noise{cfg.data.noise}.npz')
        print(f'Loading data from {data_path} ...')
        data = np.load(data_path, allow_pickle=True)
        true_data = data['data_gt']
        raw_data = data['data']
        labels = data['colors']
        train_mask = data['is_train']
        assert raw_data.shape[0] == labels.shape[0]

    log(f'Done loading raw data. Raw data shape: {raw_data.shape}', to_console=True)

    if train_mask is None:
        # no mask given, split on fly
        shuffle = cfg.training.shuffle
        train_test_split = cfg.training.train_test_split
        train_valid_split = cfg.training.train_valid_split
        if shuffle:
            idxs = np.random.permutation(len(raw_data))
            true_data = true_data[idxs]
            raw_data = raw_data[idxs]
            labels = labels[idxs]

        split_idx = int(len(raw_data)*train_test_split)
        split_val_idx = int(split_idx*train_valid_split)
        train_data = raw_data[:split_val_idx]
        train_val_data = raw_data[:split_idx]
        test_data = raw_data[split_idx:]
    else:
        # train_mask is already shuffled. Use it to split
        train_val_data = raw_data[train_mask == 1]
        split_val_idx = int(len(train_val_data)*cfg.training.train_valid_split)
        train_data = train_val_data[:split_val_idx]
        test_data = raw_data[train_mask == 0]

    print(train_mask[:10])
    print('train_val_data', train_val_data.shape, train_data.shape)
    print('split_val_idx', split_val_idx)

    train_dataset = RowStochasticDataset(data_name=cfg.data.name, X=train_data, X_labels=None, dist_type='phate_prob', knn=cfg.data.knn)
    train_val_dataset = RowStochasticDataset(data_name=cfg.data.name, X=train_val_data, X_labels=None, dist_type='phate_prob', knn=cfg.data.knn)
    whole_dataset = RowStochasticDataset(data_name=cfg.data.name, X=raw_data, X_labels=None, dist_type='phate_prob', knn=cfg.data.knn)

    log(f'Train dataset: {len(train_dataset)}; \
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
    # check if cuda is available
    device_av = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.training.accelerator is None or cfg.training.accelerator == 'auto':
        device = device_av
    else:
        device = cfg.training.accelerator
    epoch = cfg.training.max_epochs

    lr = cfg.training.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.training.weight_decay)
    early_stopper = EarlyStopping(mode='min',
                                  patience=cfg.training.patience,
                                  percentage=False)

    best_metric = np.inf
    model = model.to(device)
    log('Training ...')
    for eid in range(epoch):
        model.train()
        decoder_loss = 0.0
        encoder_loss = 0.0
        epoch_loss = 0.0
        train_Z = []
        train_indices = []
        optimizer.zero_grad()

        for (batch_inds, batch_x) in train_loader:
            batch_x = batch_x.to(device)

            # optimizer.zero_grad()
            batch_x_hat, batch_z = model(batch_x)
            train_Z.append(batch_z)
            train_indices.append(batch_inds.reshape(-1,1))

            # reconstruction loss per batch
            decoder_loss += model.decoder_loss(batch_x, batch_x_hat)
        
        # row-wise prob divergence loss
        train_Z = torch.cat(train_Z, dim=0) #[N, emb_dim]
        train_indices = torch.squeeze(torch.cat(train_indices, dim=0)) # [N,]
        pred_prob_matrix = model.compute_prob_matrix(train_Z, 
                                                     t=train_dataset.t, 
                                                     alpha=cfg.model.alpha, 
                                                     bandwidth=cfg.model.bandwidth)
        gt_prob_matrix = (train_dataset.row_stochastic_matrix).type(torch.float32).to(device)
        encoder_loss = model.encoder_loss(gt_prob_matrix, pred_prob_matrix)
        
        encoder_loss_w, decoder_loss_w = cfg.model.dist_reconstr_weights[0], cfg.model.dist_reconstr_weights[1]
        epoch_loss = encoder_loss * encoder_loss_w + (decoder_loss / len(train_loader)) * decoder_loss_w

        epoch_loss.backward()
        optimizer.step()

        if eid == 0 or eid % cfg.training.log_every_n_steps == 0:
            log(f'[Epoch: {eid}]: Encoder Loss: {encoder_loss.item()}, Decoder Loss: {decoder_loss.item()/len(train_loader)}')
            log('\r[Epoch %d] Loss: %.4f' % (eid, epoch_loss.item()))
            if run is not None:
                run.log({'train/encoder_loss': encoder_loss.item(),
                        'train/decoder_loss': decoder_loss.item()/len(train_loader),
                        'train/epoch_loss': epoch_loss.item()})

        ''' Validation '''
        model.eval()
        val_encoder_loss = 0.0
        val_decoder_loss = 0.0
        val_Z = []
        val_indices = []
        with torch.no_grad():
            for (batch_inds, batch_x) in train_val_loader:
                batch_x = batch_x.to(device)

                batch_x_hat, batch_z = model(batch_x)
                val_Z.append(batch_z)
                val_indices.append(batch_inds.reshape(-1,1)) # [B,1]
                val_decoder_loss += model.decoder_loss(batch_x, batch_x_hat).item()
            
            val_Z = torch.cat(val_Z, dim=0)
            val_indices = torch.squeeze(torch.cat(val_indices, dim=0)) # [N,]

            train_val_pred_prob_matrix = model.compute_prob_matrix(val_Z, t=train_val_dataset.t, 
                                                                   alpha=cfg.model.alpha, 
                                                                   bandwidth=cfg.model.bandwidth)
            gt_train_val_prob_matrix = (train_val_dataset.row_stochastic_matrix).type(torch.float32).to(device)
            val_encoder_loss = model.encoder_loss(gt_train_val_prob_matrix, 
                                                  train_val_pred_prob_matrix)
            
            val_loss = val_encoder_loss.item() * encoder_loss_w \
                + (val_decoder_loss / len(train_val_loader)) * decoder_loss_w
            log(f'\n[Epoch: {eid}]: Val Encoder Loss: {val_encoder_loss.item()}, Val Decoder Loss: {val_decoder_loss/len(train_val_loader)}')
            log(f'[Epoch: {eid}]: Val Loss: {val_loss}')
            if run is not None:
                run.log({'val/encoder_loss': val_encoder_loss.item(),
                            'val/decoder_loss': val_decoder_loss/len(train_val_loader),
                            'val/loss': val_loss})

        # TODO: maybe want to use a different metric for early stopping
        if val_loss < best_metric:
            if eid == 0 or eid % cfg.training.log_every_n_steps == 0:
                log('\nBetter model found. Saving best model ...\n')
            best_metric = val_loss
            best_model = model.state_dict()
            if cfg.path.save:
                # check that the directory exists if not create it
                if not os.path.exists(os.path.join(PROJECT_PATH, cfg.path.root)):
                    os.makedirs(os.path.join(PROJECT_PATH, cfg.path.root))
                torch.save(best_model, os.path.join(PROJECT_PATH, cfg.path.root, cfg.path.model))
            
        # Early Stopping
        if early_stopper.step(val_loss):
            log('Early stopping criterion met. Ending training.\n')
            break


    ''' Evaluation ''' 
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        pred_embed = model.encode(torch.from_numpy(whole_dataset.X).type(torch.float32).to(device))
        recon_data = model.decode(pred_embed).cpu().detach().numpy()
        pred_dist = model.compute_prob_matrix(pred_embed, 
                                                t=whole_dataset.t, 
                                                alpha=cfg.model.alpha, 
                                                bandwidth=cfg.model.bandwidth)

    tsne_embed = TSNE(n_components=emb_dim, perplexity=5).fit_transform(whole_dataset.X)

    # affnity matching, metrics: KL divergence, mAP
    metrics = {
        'KL div': np.inf,
        'mAP': -np.inf,
        'mAP_PHATE': -np.inf,
        'mAP_TSNE': -np.inf,
    }
    gt_dist = (whole_dataset.row_stochastic_matrix).type(torch.float32).to(device)
    metrics['KL div'] = torch.nn.functional.kl_div(torch.log(pred_dist+1e-8),
                                                    gt_dist+1e-8,
                                                    reduction='batchmean',
                                                    log_target=False).item()
    metrics['mAP'] = computeKNNmAP(pred_embed.cpu().detach().numpy(),
                                    whole_dataset.X,
                                    k=5,
                                    distance_op='norm')
    metrics['mAP_PHATE'] = computeKNNmAP(whole_dataset.phate_embed.cpu().detach().numpy(),
                                            whole_dataset.X,
                                            k=5,
                                            distance_op='norm')
    metrics['mAP_TSNE'] = computeKNNmAP(tsne_embed,
                                        whole_dataset.X,
                                        k=5,
                                        distance_op='norm')
    
    ''' DeMAP '''
    if true_data is not None:
        embedding_map = {
            'probae': pred_embed.cpu().detach().numpy(),
            'phate': whole_dataset.phate_embed.cpu().detach().numpy(),
            'tsne': tsne_embed
        }
        demaps = evaluate_demap(embedding_map, true_data)
        for k, v in demaps.items():
            metrics[f'{k}'] = v

        # Subsample on test data
        test_idx = np.nonzero(train_mask == 0)[0]
        test_embed = pred_embed[test_idx]
        demaps_test = DEMaP(true_data, test_embed, subsample_idx=test_idx)
        metrics['test'] = demaps_test

        log(f'Evaluation DeMAPs: {demaps}')

    if run is not None:
        run.log({f'evaluation/{k}': v for k, v in metrics.items()})

    ''' Visualize '''
    if labels is not None:
        labels = np.squeeze(labels)
    visualize(pred=pred_embed.cpu().detach().numpy(),
              phate_embed=whole_dataset.phate_embed.cpu().detach().numpy(),
              pred_dist=pred_dist.cpu().detach().numpy(),
              gt_dist=gt_dist.cpu().detach().numpy(),
              recon_data=recon_data,
              data=whole_dataset.X,
              dataset_name=cfg.data.name,
              data_clusters=labels,
              metrics=metrics,
              save_path=os.path.join(PROJECT_PATH, cfg.path.root, 
                                     f'{cfg.model.prob_method}_{cfg.data.name}_bw{cfg.model.bandwidth}_embeddings.png'),
              wandb_run=run)

    if run is not None:
        run.finish()


def evaluate_demap(embedding_map: dict[str, np.ndarray],
                   true_data: np.ndarray) -> dict[str, float]:
    demaps = {}
    for k, v in embedding_map.items():
        print(f'Computing DeMAP for {k} ...'
              f'embedding shape: {v.shape}, true data shape: {true_data.shape}')
        assert v.shape[0] == true_data.shape[0]
        demaps[k] = DEMaP(true_data, v)

    return demaps


def DEMaP(data, embedding, knn=10, subsample_idx=None):
    geodesic_dist = geodesic_distance(data, knn=knn)
    #geodesic_dist = compute_geodesic_distances(data, knn_geodesic=knn)
    if subsample_idx is not None:
        geodesic_dist = geodesic_dist[subsample_idx, :][:, subsample_idx]
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu()
    geodesic_dist = squareform(geodesic_dist)
    embedded_dist = pdist(embedding)
    return spearmanr(geodesic_dist, embedded_dist).correlation


def geodesic_distance(data, knn=10, distance="data"):
    G = graphtools.Graph(data, knn=knn, decay=None)
    return G.shortest_path(distance=distance)


if __name__ == "__main__":
    train_eval()