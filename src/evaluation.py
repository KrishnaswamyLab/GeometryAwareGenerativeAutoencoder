"""
Model-agnostic functions for evaluating models.
"""
import demap
from sklearn.metrics import mean_absolute_percentage_error
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
import magic

def compute_metrics(model, x_test, x_noiseless=None, dist_true=None):
    """
    model: need encode, decode functions
    x_test: test data in ambient space
    x_noiseless: noiseless data corresponding to test. only available when using synthetic data.
    dist_true: ground truth distance matrix. only used for distance matching.
    """
    # TODO use batch if the data is too large. currently not needed.
    # TODO check attr device exists
    x_tensor = torch.from_numpy(x_test).float().to(model.device)
    model.eval()
    z_pred = model.encode(x_tensor)
    x_pred = model.decode(z_pred)
    z_pred = z_pred.detach().cpu().numpy()
    x_pred = x_pred.detach().cpu().numpy()
    demap_val = np.nan
    if x_noiseless is not None:
        demap_val = demap.DEMaP(x_noiseless, x_pred)
    dist_pred = squareform(pdist(z_pred))
    acc_val = np.nan
    if dist_true is not None:
        acc_val = 1 - mean_absolute_percentage_error(dist_true, dist_pred)
    gene_corr_mse = compute_gene_corr_mse()
    result = dict(
        demap=demap_val,
        accuracy=acc_val,
        gene_corr_mse=gene_corr_mse,
    )
    return result

def rename_string(s):
    """
    Get filename for true synthetic datasets given file name of noisy dataset.
    """
    parts = s.split('_')
    parts[0] = "true"
    new_parts = parts[:-3] + parts[-1:]
    new_s = '_'.join(new_parts)    
    return new_s

def get_dataset_contents(noisy_path, noiseless_path):
    data_noisy = np.load(noisy_path, allow_pickle=True)
    X = data_noisy['data']
    train_mask = data_noisy['is_train']
    if 'dist' in data_noisy.files:
        dist = data_noisy['dist']
        dist_true=dist[~train_mask][:,~train_mask]
    else:
        dist_true=None
    data_noiseless = np.load(noiseless_path, allow_pickle=True)
    assert (train_mask == data_noiseless['is_train']).all()
    x_noiseless = data_noiseless['data'][~train_mask]
    x_test=X[~train_mask]
    return x_test, x_noiseless, dist_true

def get_data_config(filename):
    filename = filename.split('/')[-1]
    assert filename.startswith('noisy'), 'only works for noisy data!'
    parts = filename.split('_')    
    seedmethod = parts[2]+','+parts[1]
    bcv=parts[-3]
    dropout=parts[-2]
    return dict(
        seedmethod=seedmethod,
        bcv=bcv,
        dropout=dropout,
    )

def eval_results(noisy_path, noiseless_path, model):
    x_test, x_noiseless, dist_true = get_dataset_contents(noisy_path, noiseless_path)
    config = get_data_config(noisy_path)
    result = compute_metrics(model, x_test, x_noiseless, dist_true)
    for k, v in config.items():
        result[k] = v
    return result

def compute_gene_corr_mse(X_reconstructed, X_real, **kwargs):
    """
    Given two CELL x GENE matrices, computes MSE between column-wise pearson gene-gene correlations.
    """
    magic_op = magic.MAGIC(**kwargs)
    X_recon_magic = magic_op.fit_transform(X_reconstructed)
    X_real_magic = magic_op.fit_transform(X_real)
    # Compute column wise correlations within each matrix
    corrs_recon = np.corrcoef(X_recon_magic, rowvar=False) # vars (genes) are in columns
    corrs_real = np.corrcoef(X_real_magic, rowvar=False)
    # get mse
    mse = np.sum(np.square(corrs_recon - corrs_real))
    return mse
