import demap
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import pathlib

def generate_synthetic_data(nGenes=17580, batchCells=3000, nBatches=3, method='paths', bcv=0.2, dropout=0.5, seed=42):
    if method == 'paths':
        splatter = demap.splatter.paths
    elif method == 'groups':
        splatter = demap.splatter.groups
    else:
        raise ValueError('method must be one of "path" or "groups"')
    data_true = splatter(nGenes=nGenes, batchCells=batchCells, nBatches=nBatches, bcv=0, dropout=0, seed=seed)
    data_noisy = splatter(nGenes=nGenes, batchCells=batchCells, nBatches=nBatches, bcv=bcv, dropout=dropout, seed=seed)
    return data_true, data_noisy

@hydra.main(version_base=None, config_path='../data_conf', config_name='config')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    data_true, data_noisy = generate_synthetic_data(nGenes=cfg.nGenes, batchCells=cfg.batchCells, nBatches=cfg.nBatches, method=cfg.method, bcv=cfg.bcv, dropout=cfg.dropout, seed=cfg.seed)
    pathlib.Path(cfg.path).mkdir(parents=True, exist_ok=True)
    np.save(f'{cfg.path}/true_{cfg.seed}_{cfg.nGenes}_{cfg.batchCells}_{cfg.nBatches}.npy', data_true)
    np.save(f'{cfg.path}/noisy_{cfg.seed}_{cfg.nGenes}_{cfg.batchCells}_{cfg.nBatches}_{cfg.bcv}_{cfg.dropout}.npy', data_noisy)

if __name__ == "__main__":
    main()