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

def train_decoder(model, train_loader, val_loader, test_loader, cfg, save_dir, wandb_run=None):
    log_path = os.path.join(PROJECT_PATH, cfg.path.root, save_dir, cfg.path.log)
    model_save_path = os.path.join(PROJECT_PATH, cfg.path.root, save_dir, cfg.path.decoder_model)

    ''' Training '''
    device_av = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.training.accelerator
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

    log('Training Decoder ...', log_path)
    for eid in range(epoch):
        model.train()
        decoder_epoch_loss = 0.0

        for bidx, (batch_z, batch_x) in enumerate(train_loader):
            batch_z = batch_z.to(device)
            batch_x = batch_x.to(device)

            optimizer.zero_grad()


            batch_x_hat = model(batch_z)
            decoder_loss = torch.nn.functional.mse_loss(batch_x, batch_x_hat)
            decoder_epoch_loss += decoder_loss.item()

            decoder_loss.backward()
            optimizer.step()

        decoder_epoch_loss /= len(train_loader)
        log(f'[Epoch: {eid}]: Decoder Loss: {decoder_epoch_loss}')
        if wandb_run is not None:
            wandb_run.log({'train/decoder_loss': decoder_epoch_loss})

        # Validation decoder
        model.eval()
        val_decoder_loss = 0.0
        with torch.no_grad():
            for bidx, (batch_z, batch_x) in enumerate(val_loader):
                batch_z = batch_z.to(device)
                batch_x = batch_x.to(device)

                batch_x_hat = model(batch_z)
                val_decoder_loss += torch.nn.functional.mse_loss(batch_x, batch_x_hat).item()

            val_decoder_loss /= len(val_loader)
            log(f'[Epoch: {eid}]: Val Decoder Loss: {val_decoder_loss}', to_console=True)
            if wandb_run is not None:
                wandb_run.log({'val/decoder_loss': val_decoder_loss})

        if val_decoder_loss < best_metric:
            log('Better model found. Saving best model ...\n')
            best_metric = val_decoder_loss
            best_model = model.state_dict()
            if cfg.path.save:
                torch.save(best_model, model_save_path)

        # Early Stopping
        if early_stopper.step(val_decoder_loss):
            log('[Decoder] Early stopping criterion met. Ending training.\n')
            break

    log('Done training decoder.', log_path)

    ''' Evaluation '''
    model.eval()
    with torch.no_grad():
        test_decoder_loss = 0.0
        for bidx, (batch_z, batch_x) in enumerate(test_loader):
            batch_z = batch_z.to(device)
            batch_x = batch_x.to(device)

            batch_x_hat = model(batch_z)
            test_decoder_loss += torch.nn.functional.mse_loss(batch_x, batch_x_hat).item()
        test_decoder_loss /= len(test_loader)

        log(f'Test Decoder Loss: {test_decoder_loss}', log_path)
        if wandb_run is not None:
            wandb_run.log({'test/decoder_loss': test_decoder_loss})
    
    model.load_state_dict(best_model)

    return model, test_decoder_loss

def true_path_base(s):
    """
    Get filename for true synthetic datasets given base file name of noisy dataset.
    """
    parts = s.split('_')
    parts[0] = "true"
    new_parts = parts[:-3] + parts[-1:]
    new_s = '_'.join(new_parts)  
    return new_s

@hydra.main(version_base=None, config_path='../conf', config_name='separate_affinityae.yaml')
def train_eval(cfg: DictConfig):
    if cfg.model.encoding_method in ['phate', 'tsne', 'umap']:
        save_dir =  f'sepa_{cfg.model.encoding_method}_a{cfg.model.alpha}_knn{cfg.data.knn}_'
    else:
        save_dir =  f'sepa_{cfg.model.prob_method}_a{cfg.model.alpha}_knn{cfg.data.knn}_'
    if  cfg.data.name in ['splatter']:
        save_dir += f'{cfg.data.noisy_path}'
    elif cfg.data.name in ['myeloid']:
        save_dir += f'{cfg.data.name}'
    else:
        save_dir += f'{cfg.data.name}_noise{cfg.data.noise:.1f}_seed{cfg.data.seed}'
    os.makedirs(os.path.join(PROJECT_PATH, cfg.path.root, save_dir), exist_ok=True)
    
    model_save_path = os.path.join(PROJECT_PATH, cfg.path.root, save_dir, cfg.path.model)
    decoder_save_path = os.path.join(PROJECT_PATH, cfg.path.root, save_dir, cfg.path.decoder_model)
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
        splatter_data_root = '/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/synthetic_data3/'
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

    train_dataset = RowStochasticDataset(data_name=cfg.data.name, X=train_data, X_labels=None, 
                                         dist_type='phate_prob', knn=cfg.data.knn, t=cfg.data.t, n_landmark=cfg.data.n_landmark)
    train_val_dataset = RowStochasticDataset(data_name=cfg.data.name, X=train_val_data, X_labels=None, 
                                             dist_type='phate_prob', knn=cfg.data.knn, t=cfg.data.t, n_landmark=cfg.data.n_landmark)
    whole_dataset = RowStochasticDataset(data_name=cfg.data.name, X=raw_data, X_labels=None, 
                                         dist_type='phate_prob', knn=cfg.data.knn, t=cfg.data.t, n_landmark=cfg.data.n_landmark)

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
                     prob_method=cfg.model.prob_method, dist_reconstr_weights=cfg.model.dist_reconstr_weights)
    decoder = Decoder(dim=raw_data.shape[1], emb_dim=emb_dim, layer_widths=cfg.model.layer_widths[::-1], activation_fn=act_fn)

    if cfg.model.load_encoder and os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        log(f'Loaded encoder from {model_save_path}, skipping encoder training ...')
        train_encoder = False
    else:
        log('Training encoder from scratch ...')
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

    log('Training Encoder ...')
    for eid in range(epoch):
        model.train()
        encoder_loss = 0.0
        train_Z = []
        train_indices = []
        optimizer.zero_grad()

        for (batch_inds, batch_x) in train_loader:
            batch_x = batch_x.to(device)

            batch_z = model.encode(batch_x)
            train_Z.append(batch_z)
            train_indices.append(batch_inds.reshape(-1,1))
        
        # row-wise prob divergence loss
        train_Z = torch.cat(train_Z, dim=0) #[N, emb_dim]
        train_indices = torch.squeeze(torch.cat(train_indices, dim=0)) # [N,]

        pred_prob_matrix = model.compute_prob_matrix(train_Z, 
                                                     t=train_dataset.t, 
                                                     alpha=cfg.model.alpha, 
                                                     bandwidth=cfg.model.bandwidth)
        gt_prob_matrix = (train_dataset.row_stochastic_matrix).type(torch.float32).to(device)
        encoder_loss = model.encoder_loss(gt_prob_matrix, pred_prob_matrix)
    
        encoder_loss.backward()
        optimizer.step()

        if eid == 0 or eid % cfg.training.log_every_n_steps == 0:
            log(f'[Epoch: {eid}]: Encoder Loss: {encoder_loss.item()}')
            if wandb_run is not None:
                wandb_run.log({'train/encoder_loss': encoder_loss.item()})

        ''' Validation (used both train + val data to compute prob matrix)'''
        model.eval()
        val_encoder_loss = 0.0
        val_Z = []
        val_indices = []
        with torch.no_grad():
            for (batch_inds, batch_x) in train_val_loader:
                batch_x = batch_x.to(device)

                batch_z = model.encode(batch_x)
                val_Z.append(batch_z)
                val_indices.append(batch_inds.reshape(-1,1)) # [B,1]
            
            val_Z = torch.cat(val_Z, dim=0)
            val_indices = torch.squeeze(torch.cat(val_indices, dim=0)) # [N,]

            train_val_pred_prob_matrix = model.compute_prob_matrix(val_Z, t=train_val_dataset.t, 
                                                                   alpha=cfg.model.alpha, 
                                                                   bandwidth=cfg.model.bandwidth)
            gt_train_val_prob_matrix = (train_val_dataset.row_stochastic_matrix).type(torch.float32).to(device)
            val_encoder_loss = model.encoder_loss(gt_train_val_prob_matrix, 
                                                  train_val_pred_prob_matrix)
            
            log(f'\n[Epoch: {eid}]: Val Encoder Loss: {val_encoder_loss.item()}')
            if wandb_run is not None:
                wandb_run.log({'val/encoder_loss': val_encoder_loss.item()})

        # TODO: maybe want to use a different metric for early stopping
        if val_encoder_loss < best_metric:
            log('\nBetter model found. Saving best model ...\n')
            best_metric = val_encoder_loss
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
        pred_dist = model.compute_prob_matrix(pred_embed, 
                                                t=whole_dataset.t, 
                                                alpha=cfg.model.alpha, 
                                                bandwidth=cfg.model.bandwidth)

    tsne_embed = TSNE(n_components=emb_dim, perplexity=5).fit_transform(whole_dataset.X)
    umap_embed = umap.UMAP().fit_transform(whole_dataset.X)
    dm_embed = DiffusionMap().fit_transform(whole_dataset.X)

    # affnity matching, metrics: KL divergence
    metrics = {}
    gt_dist = (whole_dataset.row_stochastic_matrix).type(torch.float32).to(device)
    metrics['KL'] = torch.nn.functional.kl_div(torch.log(pred_dist+1e-8),
                                                    gt_dist+1e-8,
                                                    reduction='batchmean',
                                                    log_target=False).item()
    
    ''' DeMAP '''
    if true_data is not None and cfg.model.encoding_method == 'affinity':
        embedding_map = {
            'Affinity': pred_embed.cpu().detach().numpy(),
            '-log(Aff)': -torch.log(pred_dist+1e-8).cpu().detach().numpy(),
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

        # DeMAP(-log(aff)) on test set
        test_dist = -torch.log(pred_dist[test_idx]+1e-8).cpu().detach().numpy()
        metrics['-log(Aff)_Test'] = DEMaP(true_data, test_dist, subsample_idx=test_idx)

        log(f'Evaluation metrics: {metrics}')
        if wandb_run is not None:
            wandb_run.log({f'evaluation/{k}': v for k, v in metrics.items()})
        
        # Save metrics to npz file
        print('Saving eval metrics to ', metrics_save_path)
        np.savez(metrics_save_path, **metrics)


    log('Done training & evaluating encoder.')


    ''' Decoder '''
    log('Training decoder while keep encoder frozen...')

    # Generate frozen embeddings
    if cfg.model.encoding_method == 'phate':
        train_val_frozen = whole_dataset.phate_embed.cpu().detach().numpy()[train_mask == 1]
        train_frozen = train_val_frozen[:split_val_idx]
        val_frozen = train_val_frozen[split_val_idx:]
        test_frozen = whole_dataset.phate_embed.cpu().detach().numpy()[train_mask == 0]
    elif cfg.model.encoding_method == 'tsne':
        train_val_frozen = tsne_embed[train_mask == 1]
        train_frozen = train_val_frozen[:split_val_idx]
        val_frozen = train_val_frozen[split_val_idx:]
        test_frozen = tsne_embed[train_mask == 0]
    elif cfg.model.encoding_method == 'umap':
        train_val_frozen = umap_embed[train_mask == 1]
        train_frozen = train_val_frozen[:split_val_idx]
        val_frozen = train_val_frozen[split_val_idx:]
        test_frozen = umap_embed[train_mask == 0]
    else:
        # Use our own affinity matching encoder
        train_val_frozen = pred_embed[train_mask == 1].cpu().detach().numpy()
        train_frozen = train_val_frozen[:split_val_idx]
        val_frozen = train_val_frozen[split_val_idx:]
        test_frozen = pred_embed[train_mask == 0].cpu().detach().numpy()
        log(f'Train frozen shape: {train_frozen.shape}, {val_frozen.shape}, {test_frozen.shape}')
        log(f'data shape: {train_data.shape}, {val_data.shape}, {test_data.shape}')

    frozen_train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_frozen).float(), torch.from_numpy(train_data).float())
    frozen_val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_frozen).float(), torch.from_numpy(val_data).float())
    frozen_test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_frozen).float(), torch.from_numpy(test_data).float())

    frozen_train_loader = torch.utils.data.DataLoader(frozen_train_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    frozen_val_loader = torch.utils.data.DataLoader(frozen_val_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    frozen_test_loader = torch.utils.data.DataLoader(frozen_test_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    log('Done creating frozen datasets.')


    if cfg.model.load_decoder is True and os.path.exists(decoder_save_path):
        decoder.load_state_dict(torch.load(decoder_save_path))
        log(f'Loaded decoder from {decoder_save_path}, skipping decoder training ...')
    else:
        log('Training decoder from scratch ...')
        decoder, test_decoder_loss = train_decoder(decoder, frozen_train_loader, frozen_val_loader, frozen_test_loader, 
                                                   cfg, save_dir, wandb_run)


    ''' Reconstruction '''
    if cfg.model.encoding_method == 'affinity':
        model.eval()
        with torch.no_grad():
            pred_embed = model.encode(torch.from_numpy(whole_dataset.X).type(torch.float32).to(device))
            recon_data = decoder(pred_embed).cpu().detach().numpy()
    elif cfg.model.encoding_method == 'tsne':
        recon_data = decoder(tsne_embed).cpu().detach().numpy()

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

def DEMaP(data, embedding, knn=10, subsample_idx=None):
    geodesic_dist = geodesic_distance(data, knn=knn)
    #geodesic_dist = compute_geodesic_distances(data, knn_geodesic=knn)
    if subsample_idx is not None:
        geodesic_dist = geodesic_dist[subsample_idx, :][:, subsample_idx]
    geodesic_dist = squareform(geodesic_dist)
    embedded_dist = pdist(embedding)
    return spearmanr(geodesic_dist, embedded_dist).correlation

def geodesic_distance(data, knn=10, distance="data"):
    G = graphtools.Graph(data, knn=knn, decay=None)
    return G.shortest_path(distance=distance)


if __name__ == "__main__":
    print('os.environ:', PROJECT_PATH)

    train_eval()