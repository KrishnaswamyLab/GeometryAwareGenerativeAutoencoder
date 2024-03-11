import numpy as np
import scprep
import torch
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
from sklearn.manifold import MDS

from procrustes import Procrustes

def visualize(pred: np.ndarray,
              phate_embed: np.ndarray,
              other_embeds: dict[str, np.ndarray],
              pred_dist: np.ndarray,
              gt_dist: np.ndarray,
              recon_data: np.ndarray,
              data: np.ndarray,
              dataset_name: str,
              data_clusters: np.ndarray,
              metrics: dict = None,
              save_path: str = '../embeddings.png',
              wandb_run: wandb.sdk.wandb_run.Run = None):
    print("Visualizing Embeddings...")

    # first row has 4 plots: gt_dist vs. pred_dist, pred, data, recon_data(if given)
    # second row has 1+len(other_embeds) plots: phate_embed, other_embeds
    n_subplots = 4 + 1 + len(other_embeds)
    n_plots_per_row = 4
    n_rows = n_subplots // n_plots_per_row

    # Plot Embeddings
    plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(n_plots_per_row * 7, n_rows * 6))
    size = 5
    fontsize = 6

    # Distribution of gt_dist vs. pred_dist.
    ax = fig.add_subplot(n_rows, n_plots_per_row, 1)
    ax.hist(gt_dist.flatten(), bins=100, label='gt', color='blue')
    ax.hist(pred_dist.flatten(), bins=100, label='pred', color='gray', alpha=0.5)
    ax.legend()
    ax.title.set_text("gt vs. pred")

    # Predicted Embeddings
    ax = fig.add_subplot(n_rows, n_plots_per_row, 2)
    title_str = ''
    for metric, value in metrics.items():
        title_str += f"{metric}: {value:.2f}, "
    scprep.plot.scatter2d(pred,
                        c=data_clusters,
                        legend=False,
                        ax=ax,
                        title=title_str,
                        xticks=True,
                        yticks=True,
                        label_prefix='',
                        fontsize=fontsize,
                        s=size)
    
    # Visualize < 3D embeddings
    if data.shape[1] == 2:
        ax = fig.add_subplot(n_rows, n_plots_per_row, 3)
        scprep.plot.scatter2d(data, c=data_clusters, s=5, ax=ax)
        ax.title.set_text(dataset_name)
    elif data.shape[1] == 3:
        ax = fig.add_subplot(n_rows, n_plots_per_row, 3, projection='3d')
        scprep.plot.scatter3d(data, c=data_clusters, s=5, ax=ax)
        ax.title.set_text(dataset_name)
    
    # Visualize reconstructed data
    if recon_data is not None:
        if data.shape[1] == 2:
            ax = fig.add_subplot(n_rows, n_plots_per_row, 4)
            scprep.plot.scatter2d(recon_data, c=data_clusters, s=5, ax=ax)
            ax.title.set_text("Reconstructed Data")
        elif data.shape[1] == 3:
            ax = fig.add_subplot(n_rows, n_plots_per_row, 4, projection='3d')
            scprep.plot.scatter3d(recon_data, c=data_clusters, s=5, ax=ax)
            ax.title.set_text("Reconstructed Data")
    
    # Visualize PHATE Embeddings
    ax = fig.add_subplot(n_rows, n_plots_per_row, 5)
    scprep.plot.scatter2d(phate_embed, c=data_clusters, s=5, ax=ax)
    ax.title.set_text("PHATE")

    # Visualize other embeddings
    for i, (name, embed) in enumerate(other_embeds.items()):
        ax = fig.add_subplot(n_rows, n_plots_per_row, 6 + i)
        scprep.plot.scatter2d(embed, c=data_clusters, s=5, ax=ax)
        ax.title.set_text(name)


    plt.tight_layout()
    plt.savefig(save_path)

    print("Saved Embeddings to %s" % (save_path))

    if wandb_run is not None:
        wandb_run.log({"Embeddings": wandb.Image(save_path)})