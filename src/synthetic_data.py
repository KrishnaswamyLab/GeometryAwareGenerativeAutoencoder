import demap
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import pathlib

def generate_synthetic_data(nGenes=17580, batchCells=3000, nBatches=1, method='paths', bcv=0.2, dropout=0.5, seed=42, n_pca=100):
    if method == 'paths':
        splatter = demap.splatter.paths
    elif method == 'groups':
        splatter = demap.splatter.groups
    else:
        raise ValueError('method must be one of "path" or "groups"')
    data_true_pc, data_true = splatter(nGenes=nGenes, batchCells=batchCells, nBatches=nBatches, bcv=0, dropout=0, seed=seed, n_pca=n_pca)
    data_noisy_pc, data_noisy = splatter(nGenes=nGenes, batchCells=batchCells, nBatches=nBatches, bcv=bcv, dropout=dropout, seed=seed, n_pca=n_pca)
    return data_true_pc, data_true, data_noisy_pc, data_noisy

def convert_data(X, seed=42, test_size=0.2):
    phate_op = phate.PHATE()
    phate_data = phate_op.fit_transform(X)
    colors = np.zeros(X.shape[0])
    dists = squareform(pdist(phate_op.diff_potential))
    ids = np.arange(0, X.shape[0])
    id_train, id_test = train_test_split(ids, test_size=test_size, random_state=seed)
    is_train = np.isin(ids, id_train)
    return dict(
        data=X,
        colors=colors,
        dist=dists,
        phate=phate_data,
        is_train=is_train
    )

@hydra.main(version_base=None, config_path='../data_conf', config_name='config')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    data_true_pc, data_true, data_noisy_pc, data_noisy = generate_synthetic_data(nGenes=cfg.nGenes, batchCells=cfg.batchCells, nBatches=cfg.nBatches, method=cfg.method, bcv=cfg.bcv, dropout=cfg.dropout, seed=cfg.seed)
    pathlib.Path(cfg.path).mkdir(parents=True, exist_ok=True)
    np.save(f'{cfg.path}/true_{cfg.seed}_{cfg.method}_{cfg.nGenes}_{cfg.batchCells}_{cfg.nBatches}.npy', data_true_pc)
    np.save(f'{cfg.path}/noisy_{cfg.seed}_{cfg.method}_{cfg.nGenes}_{cfg.batchCells}_{cfg.nBatches}_{cfg.bcv}_{cfg.dropout}.npy', data_noisy_pc)
    np.save(f'{cfg.path}/nopc_true_{cfg.seed}_{cfg.method}_{cfg.nGenes}_{cfg.batchCells}_{cfg.nBatches}.npy', data_true)
    np.save(f'{cfg.path}/nopc_noisy_{cfg.seed}_{cfg.method}_{cfg.nGenes}_{cfg.batchCells}_{cfg.nBatches}_{cfg.bcv}_{cfg.dropout}.npy', data_noisy)
    data_dict = convert_data(data_noisy_pc, seed=cfg.seed, test_size=cfg.test_size)
    np.savez(f'{cfg.path}/noisy_{cfg.seed}_{cfg.method}_{cfg.nGenes}_{cfg.batchCells}_{cfg.nBatches}_{cfg.bcv}_{cfg.dropout}_all.npz', **data_dict)
    data_dict = convert_data(data_true_pc, seed=cfg.seed, test_size=cfg.test_size)
    np.savez(f'{cfg.path}/true_{cfg.seed}_{cfg.method}_{cfg.nGenes}_{cfg.batchCells}_{cfg.nBatches}_all.npz', **data_dict)

if __name__ == "__main__":
    main()