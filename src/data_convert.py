"""
Split the npy data into train and test; get phate distances to prepare for the model.
"""
from sklearn.model_selection import train_test_split
import phate
import numpy as np
from scipy.spatial.distance import pdist, squareform
from omegaconf import DictConfig, OmegaConf
import hydra

# def convert_data(X, seed=42, test_size=0.2):
#     phate_op = phate.PHATE()
#     phate_data = phate_op.fit_transform(X)
#     colors = np.zeros(X.shape[0])
#     dists = squareform(pdist(phate_op.diff_potential))
#     ids = np.arange(0, X.shape[0])
#     id_train, id_test = train_test_split(ids, test_size=test_size, random_state=seed)
#     is_train = np.isin(ids, id_train)
#     return dict(
#         data=X,
#         colors=colors,
#         dist=dists,
#         phate=phate_data,
#         is_train=is_train
#     )

# to simulate the application case, do not include test points when running phate for training points.
# but include training points when running phate for test points.
# def convert_data(X, seed=42, test_size=0.2):
#     phate_op = phate.PHATE()
#     phate_data = phate_op.fit_transform(X)
#     colors = np.zeros(X.shape[0])
#     dists_all = squareform(pdist(phate_op.diff_potential))
#     ids = np.arange(0, X.shape[0])
#     id_train, id_test = train_test_split(ids, test_size=test_size, random_state=seed)
#     is_train = np.isin(ids, id_train)
#     X_train = X[is_train]
#     phate_op = phate.PHATE()
#     _ = phate_op.fit_transform(X_train)
#     dists = dists_all.copy()
#     dists_train = squareform(pdist(phate_op.diff_potential))
#     dists[is_train][:,is_train] = dists_train
#     return dict(
#         data=X,
#         colors=colors,
#         dist=dists,
#         dist_all=dists_all, # deprecated.
#         phate=phate_data,
#         is_train=is_train
#     )

def convert_data(X, seed=42, test_size=0.2, knn=5, t=30, n_components=3, decay=40):
    # phate_op = phate.PHATE(random_state=42, knn=20, t=30, n_components=3, decay=5)
    phate_op = phate.PHATE(random_state=seed, t=t, n_components=n_components, knn=knn, decay=decay)
    phate_data = phate_op.fit_transform(X)
    colors = np.zeros(X.shape[0])
    dists_all = squareform(pdist(phate_op.diff_potential))
    ids = np.arange(0, X.shape[0])
    id_train, id_test = train_test_split(ids, test_size=test_size, random_state=seed)
    is_train = np.isin(ids, id_train)
    X_train = X[is_train]
    # phate_op = phate.PHATE(random_state=seed, knn=20, t=30, n_components=3, decay=5)
    phate_op = phate.PHATE(random_state=seed, t=t, n_components=n_components, knn=knn, decay=decay)
    _ = phate_op.fit_transform(X_train)
    dists = dists_all.copy()
    dists_train = squareform(pdist(phate_op.diff_potential))
    dists[is_train][:,is_train] = dists_train
    return dict(
        data=X,
        colors=colors,
        dist=dists,
        dist_all=dists_all, # deprecated.
        phate=phate_data,
        is_train=is_train
    )

@hydra.main(version_base=None, config_path='../data_conf', config_name='config')
def main(cfg: DictConfig):
    data_noisy = np.load(f'{cfg.path}/noisy_{cfg.seed}_{cfg.method}_{cfg.nGenes}_{cfg.batchCells}_{cfg.nBatches}_{cfg.bcv}_{cfg.dropout}.npy')
    data_dict = convert_data(data_noisy, seed=cfg.seed, test_size=cfg.test_size)
    np.savez(f'{cfg.path}/noisy_{cfg.seed}_{cfg.method}_{cfg.nGenes}_{cfg.batchCells}_{cfg.nBatches}_{cfg.bcv}_{cfg.dropout}_all.npz', **data_dict)
    data_true = np.load(f'{cfg.path}/true_{cfg.seed}_{cfg.method}_{cfg.nGenes}_{cfg.batchCells}_{cfg.nBatches}.npy')
    data_dict = convert_data(data_true, seed=cfg.seed, test_size=cfg.test_size)
    np.savez(f'{cfg.path}/true_{cfg.seed}_{cfg.method}_{cfg.nGenes}_{cfg.batchCells}_{cfg.nBatches}_all.npz', **data_dict)

if __name__ == "__main__":
    main()