"""
Model-agnostic functions for evaluating models.
"""
import demap
from sklearn.metrics import mean_absolute_percentage_error
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import magic
from sklearn.decomposition import PCA

def compute_encoding_metrics(model, x_test, x_noiseless=None, dist_true=None):
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
    z_pred = model.encoder(x_tensor)
    x_pred = model.decoder(z_pred)
    z_pred = z_pred.detach().cpu().numpy()
    x_pred = x_pred.detach().cpu().numpy()
    demap_val = np.nan
    if x_noiseless is not None:
        demap_val = demap.DEMaP(x_noiseless, z_pred)
    dist_pred = squareform(pdist(z_pred))
    acc_val = np.nan
    if dist_true is not None:
        acc_val = 1 - mean_absolute_percentage_error(dist_true, dist_pred)
    result = dict(
        demap=demap_val,
        accuracy=acc_val,
    )
    return result

def get_noiseless_name(s):
    """
    Get filename for true synthetic datasets given file name of noisy dataset.
    """
    parts = s.split('_')
    parts[0] = "true"
    new_parts = parts[:-3] + parts[-1:]
    new_s = '_'.join(new_parts)    
    return new_s

def get_ambient_name(s):
    return 'nopc_'+ s[:-4]

def get_dataset_contents(noisy_path, noiseless_path, ambient_path=None):
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
    pca = None
    if ambient_path is not None:
        data_ambient = np.load(ambient_path, allow_pickle=True)
        pca = PCA(n_components=x_test.shape[1])
        x_test_pc = pca.fit_transform(data_ambient)
    return x_test, x_noiseless, dist_true, pca

def get_dataset_contents_w_gt(data_path):
    data_all = np.load(data_path, allow_pickle=True)
    X = data_all['data']
    train_mask = data_all['is_train']
    if 'dist' in data_all.files:
        dist = data_all['dist']
        dist_true=dist[~train_mask][:,~train_mask]
    else:
        dist_true=None
    # data_noiseless = np.load(noiseless_path, allow_pickle=True)
    # assert (train_mask == data_noiseless['is_train']).all()
    # x_noiseless = data_noiseless['data'][~train_mask]
    x_noiseless = data_all['data_gt'][~train_mask]
    x_test=X[~train_mask]
    class dummyPCA:
        def transform(self, x):
            return x
        def inverse_transform(self, x):
            return x
    pca = dummyPCA()
    # pca = None
    # if ambient_path is not None:
    #     data_ambient = np.load(ambient_path, allow_pickle=True)
    #     pca = PCA(n_components=x_test.shape[1])
    #     x_test_pc = pca.fit_transform(data_ambient)
    return x_test, x_noiseless, dist_true, pca

def get_dataset_all(noisy_path, noiseless_path, ambient_path=None):
    data_noisy = np.load(noisy_path, allow_pickle=True)
    X = data_noisy['data']
    train_mask = data_noisy['is_train']
    if 'dist' in data_noisy.files:
        dist = data_noisy['dist']
        dist_true=dist
    else:
        dist_true=None
    data_noiseless = np.load(noiseless_path, allow_pickle=True)
    assert (train_mask == data_noiseless['is_train']).all()
    x_noiseless = data_noiseless['data']
    x_all=X
    pca = None
    if ambient_path is not None:
        data_ambient = np.load(ambient_path, allow_pickle=True)
        pca = PCA(n_components=x_all.shape[1])
        x_all_pc = pca.fit_transform(data_ambient)
    return x_all, x_noiseless, dist_true, pca, train_mask

def get_data_config(filename):
    filename = filename.split('/')[-1]
    assert filename.startswith('noisy'), 'only works for noisy data!'
    parts = filename.split('_')    
    seed = parts[2]
    method = parts[1]
    # seedmethod = parts[2]+','+parts[1]
    bcv=parts[-3]
    dropout=parts[-2]
    return dict(
        seed=seed,
        method=method,
        # seedmethod=seedmethod,
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
    # we probably wouldn't run MAGIC on reconstructed data because it is already denoised.
    # X_recon_magic = magic_op.fit_transform(X_reconstructed)
    X_recon_magic = X_reconstructed
    X_real_magic = magic_op.fit_transform(X_real)
    # Compute column wise correlations within each matrix
    corrs_recon = np.corrcoef(X_recon_magic, rowvar=False) # vars (genes) are in columns
    corrs_real = np.corrcoef(X_real_magic, rowvar=False)
    # get mse
    mse = np.mean(np.square(corrs_recon - corrs_real))
    return mse

def spearman_correlation_corresponding(A, B):
    """
    Compute the Spearman correlation for each corresponding column of matrices A and B using a loop.
    """
    assert A.shape == B.shape, "Matrices A and B must have the same size"
    correlations = []
    for i in range(A.shape[1]):
        corr, _ = spearmanr(A[:, i], B[:, i])
        correlations.append(corr)
    return np.array(correlations)

def pearson_correlation_corresponding(A, B):
    """
    Compute the Spearman correlation for each corresponding column of matrices A and B using a loop.
    """
    assert A.shape == B.shape, "Matrices A and B must have the same size"
    correlations = []
    for i in range(A.shape[1]):
        corr = np.corrcoef(A[:, i], B[:, i])[0,1]
        correlations.append(corr)
    return np.array(correlations)


def compute_recon_metric(model, x_test, pca):
    x_tensor = torch.from_numpy(x_test).float().to(model.device)
    model.eval()
    z_pred = model.encoder(x_tensor)
    x_pred = model.decoder(z_pred)
    z_pred = z_pred.detach().cpu().numpy()
    x_pred = x_pred.detach().cpu().numpy()
    magic_op = magic.MAGIC(verbose=0)
    x_magic = magic_op.fit_transform(x_test)
    x_pred_ambient = pca.inverse_transform(x_pred)
    x_magic_ambient = pca.inverse_transform(x_magic)
    # corrs = spearman_correlation_corresponding(x_pred_ambient, x_magic_ambient)
    corrs = pearson_correlation_corresponding(x_pred_ambient, x_magic_ambient)
    score = corrs.mean()
    n_top_genes = 1000
    orig_vars = x_magic_ambient.var(axis=0)
    top_100_idx = np.argsort(orig_vars)[::-1][:n_top_genes]
    magic_top100 = x_magic_ambient[:,top_100_idx]
    dec_top100 = x_pred_ambient[:,top_100_idx]
    corrs_magic = np.corrcoef(magic_top100, rowvar=False)
    corrs_dec = np.corrcoef(dec_top100, rowvar=False)
    score2 = ((corrs_magic - corrs_dec)**2).mean()
    return score, score2

def compute_all_metrics(model, data_path, noiseless_path, ambient_path, w_gt=False):
    if w_gt:
        x_test, x_noiseless, dist_true, pca = get_dataset_contents_w_gt(data_path) # ignore the noiseless path
    else:
        x_test, x_noiseless, dist_true, pca = get_dataset_contents(data_path, noiseless_path, ambient_path)
    
    # Only run recon score if model has decode function
    if hasattr(model, 'decoder'):
        score1, score2 = compute_recon_metric(model, x_test, pca)
    else:
        score1, score2 = np.nan, np.nan

    encoding_metrics = compute_encoding_metrics(model, x_test, x_noiseless, dist_true)
    if w_gt:
        res_dict = dict(
            data=data_path
        )
    else:
        res_dict = get_data_config(data_path)

    for k, v in encoding_metrics.items():
        res_dict[k] = v
    res_dict['DRS'] = score1
    res_dict['DGCS'] = score2
    # res_dict['reconstr_weight'] = model.reconstr_weight
    return res_dict