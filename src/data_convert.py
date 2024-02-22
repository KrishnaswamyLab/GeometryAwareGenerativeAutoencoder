"""
Split the npy data into train and test; get phate distances to prepare for the model.
"""
from sklearn.model_selection import train_test_split
import phate
import numpy as np
from scipy.spatial.distance import pdist, squareform
from omegaconf import DictConfig, OmegaConf
import hydra

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
    data_noisy = np.load(f'{cfg.path}/noisy_{cfg.seed}_{cfg.method}_{cfg.nGenes}_{cfg.batchCells}_{cfg.nBatches}_{cfg.bcv}_{cfg.dropout}.npy')
    data_dict = convert_data(data_noisy, seed=cfg.seed, test_size=cfg.test_size)
    np.savez(f'{cfg.path}/noisy_{cfg.seed}_{cfg.method}_{cfg.nGenes}_{cfg.batchCells}_{cfg.nBatches}_{cfg.bcv}_{cfg.dropout}_all.npz', **data_dict)
    data_true = np.load(f'{cfg.path}/true_{cfg.seed}_{cfg.method}_{cfg.nGenes}_{cfg.batchCells}_{cfg.nBatches}.npy')
    data_dict = convert_data(data_true, seed=cfg.seed, test_size=cfg.test_size)
    np.savez(f'{cfg.path}/true_{cfg.seed}_{cfg.method}_{cfg.nGenes}_{cfg.batchCells}_{cfg.nBatches}_all.npz', **data_dict)

if __name__ == "__main__":
    main()