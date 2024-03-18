from typing import Tuple
import os
import matplotlib.pyplot as plt
import wandb
from dotenv import load_dotenv
import hydra

import numpy as np
import pandas as pd
import torch
import graphtools
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
import umap
from other_methods import DiffusionMap
from omegaconf import DictConfig, OmegaConf

from data import RowStochasticDataset
from model import AEProb, Decoder
from metrics import distance_distortion, mAP, computeKNNmAP
from utils.early_stop import EarlyStopping
from utils.log_utils import log
from utils.seed import seed_everything
from visualize import visualize

load_dotenv('../.env')
PROJECT_PATH=os.getenv('PROJECT_PATH')

def true_path_base(s):
    """
    Get filename for true synthetic datasets given base file name of noisy dataset.
    """
    parts = s.split('_')
    parts[0] = "true"
    new_parts = parts[:-3] + parts[-1:]
    new_s = '_'.join(new_parts)  
    return new_s

@hydra.main(version_base=None, config_path='../conf', config_name='distance_matching.yaml')
def train_eval(cfg: DictConfig):
    if cfg.model.encoding_method in ['phate', 'tsne', 'umap']:
        save_dir =  f'sepa_{cfg.model.encoding_method}_a{cfg.model.alpha}_knn{cfg.data.knn}_'
    else:
        save_dir =  f'sepa_{cfg.model.encoding_method}_a{cfg.model.alpha}_knn{cfg.data.knn}_'
    if  cfg.data.name in ['splatter']:
        save_dir += f'{cfg.data.noisy_path.split(".")[0]}'
    elif cfg.data.name in ['myeloid']:
        save_dir += f'{cfg.data.name}'
    else:
        save_dir += f'{cfg.data.name}_noise{cfg.data.noise:.1f}_seed{cfg.data.seed}'
    os.makedirs(os.path.join(PROJECT_PATH, cfg.path.root, save_dir), exist_ok=True)
    
    model_save_path = os.path.join(PROJECT_PATH, cfg.path.root, save_dir, cfg.path.model)
    visualization_save_path = os.path.join(PROJECT_PATH, cfg.path.root, save_dir, 'embeddings.png')
    metrics_save_path = os.path.join(PROJECT_PATH, cfg.path.root, save_dir, 'metrics.npz')

    if cfg.logger.use_wandb:
        config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_run = wandb.init(
            entity=cfg.logger.entity,
            project=cfg.logger.project,
            tags=cfg.logger.tags,
            name=save_dir,
            reinit=True,
            config=config,
            settings=wandb.Settings(start_method="thread"),
        )
    else:
        wandb_run = None
    log(cfg)

    # Seed everything
    print('seed everything.')
    seed_everything(cfg.training.seed)

    ''' Data '''
    if cfg.data.name == 'splatter':
        splatter_data_root = '/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/synthetic_data/'
        noisy_data_path = os.path.join(splatter_data_root, cfg.data.noisy_path)
        print('Loading data from', noisy_data_path, '...')
        true_data_path = os.path.join(splatter_data_root, true_path_base(os.path.basename(noisy_data_path)))
        print('Loading true data from', true_data_path, '...')
        noise_data = np.load(noisy_data_path)
        true_data = np.load(true_data_path)
        true_data = true_data['data']
        raw_data = noise_data['data']
        labels = noise_data['colors']
        train_mask = noise_data['is_train']
        if 'bool' in train_mask.dtype.name:
            train_mask = train_mask.astype(int)
    elif cfg.data.name in ['myeloid', 'eb', 'sea_ad']:
        data_path = os.path.join(PROJECT_PATH, cfg.data.root, f'{cfg.data.name}.npz')
        print(f'Loading data from {data_path} ...')
        data = np.load(data_path, allow_pickle=True)
        true_data = None
        raw_data = data['data']
        labels = None
        train_mask = data['is_train']
    else: 
        data_path = os.path.join(PROJECT_PATH, cfg.data.root, f'{cfg.data.name}_noise{cfg.data.noise}_seed{cfg.data.seed}.npz')
        print(f'Loading data from {data_path} ...')
        data = np.load(data_path, allow_pickle=True)
        true_data = data['data_gt']
        raw_data = data['data']
        labels = data['colors']
        train_mask = data['is_train']

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
        val_data = train_val_data[split_val_idx:]
        test_data = raw_data[train_mask == 0]

    train_dataset = RowStochasticDataset(data_name=cfg.data.name, X=train_data, X_labels=None, dist_type='phate_dist',
                                          dist_normalization=cfg.model.dist_normalization, 
                                          knn=cfg.data.knn, t=cfg.data.t)
    train_val_dataset = RowStochasticDataset(data_name=cfg.data.name, X=train_val_data, X_labels=None, dist_type='phate_dist', 
                                             knn=cfg.data.knn, t=cfg.data.t)
    whole_dataset = RowStochasticDataset(data_name=cfg.data.name, X=raw_data, X_labels=None, dist_type='phate_dist', 
                                         knn=cfg.data.knn, t=cfg.data.t)

    log(f'Train dataset: {len(train_dataset)}; \
          Val dataset: {len(train_val_dataset)}; \
          Whole dataset: {len(whole_dataset)}')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=cfg.training.batch_size, shuffle=False)

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
                     prob_method=None, dist_reconstr_weights=cfg.model.dist_reconstr_weights)

    if cfg.model.load_encoder and os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        log(f'Loaded model from {model_save_path}, skipping encoder training ...')
        train_encoder = False
    else:
        log('Training model from scratch ...')
        train_encoder = True
    
    if cfg.model.encoding_method in ['phate', 'tsne', 'umap']:
        train_encoder = False
        log(f'Using {cfg.model.encoding_method} embeddings as input to decoder ...')

    ''' Training '''
    device_av = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.training.accelerator is None or cfg.training.accelerator == 'auto':
        device = device_av
    else:
        device = cfg.training.accelerator

    epoch = cfg.training.max_epochs if train_encoder else 0

    lr = cfg.training.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.training.weight_decay)
    early_stopper = EarlyStopping(mode='min',
                                  patience=cfg.training.patience,
                                  percentage=False)

    best_metric = np.inf
    model = model.to(device)

    log('Training Model ...')
    for eid in range(epoch):
        model.train()
        encoder_loss = 0.0
        decoder_loss = 0.0

        optimizer.zero_grad()

        for (batch_inds, batch_x) in train_loader:
            batch_x = batch_x.to(device)

            batch_z = model.encode(batch_x)
            x_hat = model.decode(batch_z)

            batch_gt_dist = torch.from_numpy(train_dataset.get_gt_dist(batch_inds)).type(torch.float32).to(device)
            batch_pred_dist = torch.cdist(batch_z, batch_z)

            encoder_loss = model.encoder_loss(batch_gt_dist, batch_pred_dist, type='mse')
            decoder_loss = model.decoder_loss(batch_x, x_hat)
            el_w = cfg.model.dist_reconstr_weights[0]
            dl_w = cfg.model.dist_reconstr_weights[1]
            loss = encoder_loss * el_w + decoder_loss * dl_w
    
            loss.backward()
            optimizer.step()

        if eid == 0 or eid % cfg.training.log_every_n_epochs == 0:
            log(f'[Epoch: {eid}]: Encoder Loss: {encoder_loss.item()}, Decoder Loss: {decoder_loss.item()}, Total Loss: {loss.item()}')
            if wandb_run is not None:
                wandb_run.log({'train/encoder_loss': encoder_loss.item()})
                wandb_run.log({'train/decoder_loss': decoder_loss.item()})
                wandb_run.log({'train/loss': loss.item()})

        ''' Validation (used both train + val data)'''
        model.eval()
        val_encoder_loss = 0.0
        val_decoder_loss = 0.0
        with torch.no_grad():
            for (batch_inds, batch_x) in train_val_loader:
                batch_x = batch_x.to(device)

                batch_z = model.encode(batch_x)
                x_hat = model.decode(batch_z)

                batch_gt_dist = torch.from_numpy(train_val_dataset.get_gt_dist(batch_inds)).type(torch.float32).to(device)
                batch_pred_dist = torch.cdist(batch_z, batch_z)

                val_encoder_loss = model.encoder_loss(batch_gt_dist, batch_pred_dist, type='mse')
                val_decoder_loss = model.decoder_loss(batch_x, x_hat)

                val_loss = val_encoder_loss * el_w + val_decoder_loss * dl_w
            
            log(f'\n[Epoch: {eid}]: Val Encoder Loss: {val_encoder_loss.item()}, Val Decoder Loss: {val_decoder_loss.item()}, Val Total Loss: {val_loss.item()}')
            if wandb_run is not None:
                wandb_run.log({'val/encoder_loss': val_encoder_loss.item()})
                wandb_run.log({'val/decoder_loss': val_decoder_loss.item()})
                wandb_run.log({'val/loss': val_loss.item()})

        # TODO: maybe want to use a different metric for early stopping
        if val_loss < best_metric:
            log('\nBetter model found. Saving best model ...\n')
            best_metric = val_loss
            best_model = model.state_dict()
            if cfg.path.save:
                torch.save(best_model, os.path.join(PROJECT_PATH, cfg.path.root, save_dir, cfg.path.model))
            
        # Early Stopping
        if early_stopper.step(val_encoder_loss):
            log('[Encoder] Early stopping criterion met. Ending training.\n')
            break
    
    ''' Evaluation '''
    if train_encoder is True:
        model.load_state_dict(best_model) # load best encoder for evaluation
    
    model.eval()
    with torch.no_grad():
        pred_embed = model.encode(torch.from_numpy(whole_dataset.X).type(torch.float32).to(device))
        pred_dist = torch.cdist(pred_embed, pred_embed)

    tsne_embed = TSNE(n_components=emb_dim, perplexity=5).fit_transform(whole_dataset.X)
    umap_embed = umap.UMAP().fit_transform(whole_dataset.X)
    dm_embed = DiffusionMap().fit_transform(whole_dataset.X)

    # dist matching, metrics: 
    metrics = {}
    #gt_dist = (whole_dataset.row_stochastic_matrix).type(torch.float32).to(device)
    gt_dist = torch.from_numpy(whole_dataset.gt_dist).type(torch.float32).to(device)
    # metrics['KL'] = torch.nn.functional.kl_div(torch.log(pred_dist+1e-8),
    #                                                 gt_dist+1e-8,
    #                                                 reduction='batchmean',
    #                                                 log_target=False).item()
    
    ''' DeMAP '''
    if true_data is not None and cfg.model.encoding_method == 'distance':
        embedding_map = {
            'Dist': pred_embed.cpu().detach().numpy(),
            'PHATE': whole_dataset.phate_embed.cpu().detach().numpy(),
            'TSNE': tsne_embed,
            'UMAP': umap_embed,
            'DiffMap': dm_embed,
        }
        demaps = evaluate_demap(embedding_map, true_data) # FIXME
        for k, v in demaps.items():
            metrics[f'{k}'] = v

        # DeMAP on test set
        test_idx = np.nonzero(train_mask == 0)[0]
        test_embed = pred_embed[test_idx].cpu().detach().numpy()
        demaps_test = DEMaP(true_data, test_embed, subsample_idx=test_idx)
        metrics['Test'] = demaps_test

        log(f'Evaluation metrics: {metrics}')
        if wandb_run is not None:
            wandb_run.log({f'evaluation/{k}': v for k, v in metrics.items()})
        
        # Save metrics to npz file
        print('Saving eval metrics to ', metrics_save_path)
        np.savez(metrics_save_path, **metrics)


    ''' Reconstruction '''
    if cfg.model.encoding_method == 'distance':
        model.eval()
        with torch.no_grad():
            pred_embed = model.encode(torch.from_numpy(whole_dataset.X).type(torch.float32).to(device))
            recon_data = model.decode(pred_embed).cpu().detach().numpy()

    ''' Visualize '''
    if labels is not None:
        labels = np.squeeze(labels)
    visualize(pred=pred_embed.cpu().detach().numpy(),
              phate_embed=whole_dataset.phate_embed.cpu().detach().numpy(),
              other_embeds={'TSNE': tsne_embed, 'UMAP': umap_embed, 'DiffMap': dm_embed},
              pred_dist=pred_dist.cpu().detach().numpy(),
              gt_dist=gt_dist.cpu().detach().numpy(),
              recon_data=recon_data,
              data=whole_dataset.X,
              dataset_name=cfg.data.name,
              data_clusters=labels,
              metrics=metrics,
              save_path=visualization_save_path,
              wandb_run=wandb_run)

    if cfg.logger.use_wandb:
        wandb_run.finish()


def evaluate_demap(embedding_map: dict[str, np.ndarray],
                   true_data: np.ndarray) -> dict[str, float]:
    demaps = {}
    for k, v in embedding_map.items():
        print(f'Computing DeMAP for {k} ...'
              f'embedding shape: {v.shape}, true data shape: {true_data.shape}')
        assert v.shape[0] == true_data.shape[0]
        demaps[k] = DEMaP(true_data, v)

    return demaps

def DEMaP(data, embedding, knn=30, subsample_idx=None):
    geodesic_dist = geodesic_distance(data, knn=knn)
    #geodesic_dist = compute_geodesic_distances(data, knn_geodesic=knn)
    if subsample_idx is not None:
        geodesic_dist = geodesic_dist[subsample_idx, :][:, subsample_idx]
    geodesic_dist = squareform(geodesic_dist)
    embedded_dist = pdist(embedding)
    return spearmanr(geodesic_dist, embedded_dist).correlation

def geodesic_distance(data, knn=30, distance="data"):
    G = graphtools.Graph(data, knn=knn, decay=None)
    print('yo: ', knn)
    return G.shortest_path(distance=distance)


if __name__ == "__main__":
    print('os.environ:', PROJECT_PATH)

    train_eval()