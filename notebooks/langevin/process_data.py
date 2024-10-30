import numpy as np
import matplotlib.pyplot as plt
import scprep
import phate
from heatgeo.embedding import HeatGeo
from scipy.spatial.distance import squareform, pdist
import argparse
from pathlib import Path

def process_data(input_file, method='phate', n_train=3000, savepath='.'):
    # Create savepath if it doesn't exist
    Path(savepath).mkdir(parents=True, exist_ok=True)

    # Load data
    data = np.load(input_file)
    name = input_file.split('.')[0]  # Remove file extension for output name

    # Choose embedding method
    if method.lower() == 'phate':
        phate_op = phate.PHATE(n_components=2, n_jobs=-1, knn=10, random_state=42)
        X_phate = phate_op.fit_transform(data)
        dists = squareform(pdist(X_phate))
    elif method.lower() == 'heatgeo':
        emb_op = HeatGeo(knn=5)
        X_phate = emb_op.fit_transform(data)
        emb_op.metric_computation(data)
        dists = emb_op.dist
    else:
        raise ValueError("Method must be either 'phate' or 'heatgeo'")

    # Plot results
    scprep.plot.scatter3d(data)
    plt.savefig(Path(savepath) / f'{name}_3d.png')
    plt.close()

    scprep.plot.scatter2d(X_phate)
    plt.savefig(Path(savepath) / f'{name}_2d.png')
    plt.close()

    # Prepare data for saving
    X_pca = data
    np.random.seed(32)
    if data.shape[0] >= n_train:
        train_ids = np.random.choice(data.shape[0], n_train, replace=False)
        is_train = np.zeros(data.shape[0], dtype=bool)
        is_train[train_ids] = True
    else:
        is_train = np.ones(data.shape[0], dtype=bool)

    data_dict = dict(
        data=X_pca,
        phate=X_phate,
        dist=dists,
        is_train=is_train,
    )

    # Save results
    np.savez(Path(savepath) / f'{name}_{method}.npz', **data_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data using PHATE or HeatGeo")
    parser.add_argument("input_file", help="Input .npy file")
    parser.add_argument("--method", choices=['phate', 'heatgeo'], default='phate', help="Embedding method (default: phate)")
    parser.add_argument("--n_train", type=int, default=3000, help="Number of training samples (default: 3000)")
    parser.add_argument("--savepath", default='./data/', help="Path to save output files (default: current directory)")

    args = parser.parse_args()

    process_data(args.input_file, args.method, args.n_train, args.savepath)