'''
    Unified evaluation script for all models
    Data: dict
    - data
    - colors
    - dist
    - is_train
    - data_gt
    - metadata (e.g. parameters for data generation)
'''
import os
from glob import glob
import argparse

import graphtools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from model import AEDist, AEProb

def DEMaP(data, embedding, knn=10, subsample_idx=None) -> float:
    geodesic_dist = geodesic_distance(data, knn=knn)
    if subsample_idx is not None:
        geodesic_dist = geodesic_dist[subsample_idx, :][:, subsample_idx]
    geodesic_dist = squareform(geodesic_dist)
    embedded_dist = pdist(embedding)

    return spearmanr(geodesic_dist, embedded_dist).correlation

def geodesic_distance(data, knn=10, distance="data") -> np.ndarray:
    G = graphtools.Graph(data, knn=knn, decay=None)

    return G.shortest_path(distance=distance)

def evaluate_demap(model: nn.Module, data: dict, knn: int, subsample_idx=None) -> float:
    data_gt = data['data_gt']
    if data_gt is None:
        raise ValueError("Ground truth data not found in data dictionary")
    
    # print out model methods
    #print(dir(model))
    assert hasattr(model, 'encode'), "Model must have encode method"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_data = data['data'].astype(np.float32)
    loader = torch.utils.data.DataLoader(input_data, batch_size=32, shuffle=False)

    model.to(device)
    model.eval()

    X = []
    Z = []
    for _, x in enumerate(loader):
        x = x.to(device)
        z = model.encode(x).detach().cpu().numpy()
        X.append(x)
        Z.append(z)
    X = np.concatenate(X, axis=0)
    Z = np.concatenate(Z, axis=0)
    assert X.shape[0] == Z.shape[0], "Input and latent space must have same number of samples"

    # Calculate DEMaP score
    demap = DEMaP(data_gt, Z, knn=knn, subsample_idx=None)
    
    return demap

def true_path_base(s):
    """
    Get filename for true synthetic datasets given base file name of noisy dataset.
    """
    parts = s.split('_')
    parts[0] = "true"
    new_parts = parts[:-3] + parts[-1:]
    new_s = '_'.join(new_parts)  
    return new_s

def _load_synthetic_data(data_path, synthetic_path='/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/synthetic_data'):
    '''
    Returns:
    - noise_path_2_data: dict
        - key: path to noisy data
        - value: dict
            - data: noisy data
            - data_gt: ground truth data
            - colors: colors of the data
    '''

    noise_paths = [data_path]
    true_paths = [os.path.join(synthetic_path, \
                               true_path_base(os.path.basename(s))) for s in noise_paths]
    noise_path_2_data = {}
    print(f'Loading {len(noise_paths)} synthetic datasets...')
    for noise_path, true_path in zip(noise_paths, true_paths):
        noise_data = np.load(noise_path)
        true_data = np.load(true_path)
        data = {'data': noise_data['data'], 'data_gt': true_data['data'], 'colors': noise_data['colors']}
        noise_path_2_data[noise_path] = data
    
    return noise_path_2_data

def _load_toy_data():
    pass

def _load_bio_data():
    pass

def evaluate(model, model_path: str, data_name:str, data_path: str, demap_knn: int = 10):
    # Load model
    model.load_state_dict(torch.load(model_path))
    print(model)

    print(f'Loaded model from {model_path}.')

    # Load data
    print(f'Loading data from {data_name}...')
    
    # Merge data dictionaries from different sources
    path2data = {}
    if data_name == 'synthetic':
        path2data.update(_load_synthetic_data(data_path))
    elif data_name == 'toy':
        path2data.update(_load_toy_data())
    elif data_name == 'bio':
        path2data.update(_load_bio_data())
    else:
        raise ValueError(f'Unknown data source: {data_name}')
    print(f'Loaded {len(path2data)} datasets. Evaluating...')

    # Evaluate model
    path2score = {}
    for path, data in path2data.items():
        score = evaluate_demap(model, data, knn=demap_knn)
        path2score[path] = score
        print(f'Finished evaluating {path}. Score: {score}')
    
    # Save scores to pandas dataframe
    df = pd.DataFrame(path2score.items(), columns=['path', 'score'])

    # Save to csv
    df.to_csv('evaluation.csv', index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model on synthetic, toy, and bio datasets')
    parser.add_argument('--model', type=str, default='AEProb', help='Model type to evaluate')
    parser.add_argument('--model_path', type=str, help='Path to model')
    parser.add_argument('--data_path', type=str, help='Path to data', default='/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/synthetic_data/noisy_42_groups_17580_2000_3_0.2_0.2_all.npz')
    parser.add_argument('--demap_knn', type=int, default=10, help='Number of nearest neighbors for DEMaP')
    parser.add_argument('--data_name', type=str, default='synthetic', help='data sources to evaluate on')

    args = parser.parse_args()
    args.model_path = '/gpfs/gibbs/pi/krishnaswamy_smita/dl2282/dmae/results/sepa_gaussian_splatter_bw1_knn5/model.ckpt'
    if args.data_name == 'synthetic':
        dim = 100
        emb_dim = 2
    elif args.data_name == 'toy':
        dim = 3
        emb_dim = 2
    else:
        raise ValueError(f'Unknown data source: {args.data_name}')
    
    if args.model == 'AEDist':
        raise NotImplementedError #FIXME
    elif args.model == 'AEProb':
        layer_widths = [256, 128, 64]
        act_fn = nn.ReLU()
        prob_method = 'gaussian'
        dist_reconstr_weights = [1, 0]
        model = AEProb(dim=dim, emb_dim=emb_dim, 
                        layer_widths=layer_widths, activation_fn=act_fn,
                        prob_method=prob_method, dist_reconstr_weights=dist_reconstr_weights)
    evaluate(model=model, 
             model_path=args.model_path,
             data_name=args.data_name,
             data_path=args.data_path,
             demap_knn=args.demap_knn)
    


    