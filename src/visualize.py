import numpy as np
import scprep
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
from sklearn.manifold import MDS

from procrustes import Procrustes

def visualize(pred: np.ndarray,
              phate_embed: np.ndarray,
              pred_dist: np.ndarray,
              gt_dist: np.ndarray,
              data: np.ndarray,
              dataset_name: str,
              data_clusters: np.ndarray,
              metrics: dict = None,
              save_path: str = '../embeddings.png'):
    print("Visualizing Embeddings...")

    n_subplots = 3
    if data.shape[1] <= 3:
        n_subplots = 4

    # Plot Embeddings
    plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(28,6))
    size = 5
    fontsize = 6

    # Distribution of gt_dist vs. pred_dist.
    ax = fig.add_subplot(1,n_subplots,1)
    ax.hist(gt_dist.flatten(), bins=100, label='gt_dist', color='blue')
    ax.hist(pred_dist.flatten(), bins=100, label='pred_dist', color='gray', alpha=0.5)
    ax.legend()
    ax.title.set_text("gt vs. pred")

    # PHATE Embeddings
    ax = fig.add_subplot(1,n_subplots,2)
    print(phate_embed.shape)
    phate_embed = phate_embed
    scprep.plot.scatter2d(phate_embed,
                        c=data_clusters,
                        legend=False,
                        ax=ax,
                        title="%s Phate Embedding" % (dataset_name),
                        xticks=True,
                        yticks=True,
                        label_prefix='',
                        fontsize=fontsize,
                        s=size)

    # Predicted Embeddings
    ax = fig.add_subplot(1,n_subplots,3)
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
    if data.shape[1] <=3:
        if data.shape[1] == 2:
            ax = fig.add_subplot(1,n_subplots,4)
            scprep.plot.scatter2d(data, c=data_clusters, s=5, ax=ax)
            ax.title.set_text(dataset_name)
        elif data.shape[1] == 3:
            ax = fig.add_subplot(1,n_subplots,4, projection='3d')
            scprep.plot.scatter3d(data, c=data_clusters, s=5, ax=ax)
            ax.title.set_text(dataset_name)

    # Procrustes aligned embeddings
    # procrustes_op = Procrustes()
    # _, pred_aligned, _ = procrustes_op.fit_transform(phate_embed, 
    #                                               pred)
    # ax = fig.add_subplot(1,4,4)
    # scprep.plot.scatter2d(pred_aligned,
    #                     c=data_clusters,
    #                     legend=True,
    #                     ax=ax,
    #                     title="Procrustes Aligned",
    #                     xticks=True,
    #                     yticks=True,
    #                     label_prefix='',
    #                     fontsize=fontsize,
    #                     s=size)


    plt.tight_layout()
    plt.savefig(save_path)

    print("Saved Embeddings to %s" % (save_path))