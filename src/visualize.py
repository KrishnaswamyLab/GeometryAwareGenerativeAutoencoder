import numpy as np
import scprep
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
from sklearn.manifold import MDS

from procrustes import Procrustes

def visualize(model: torch.nn.Module,
              dataset_name: str,
              data: torch.Tensor, 
              data_clusters: np.ndarray,
              phate_embed: torch.Tensor, 
              gt_dist: torch.Tensor, 
              dist_type: str = 'prob',
              metrics: dict = None,
              save_path: str = '../embeddings.png'):
    print("Visualizing Embeddings...")

    model.eval()
    pred = model.encode(data)

    # Plot Embeddings
    plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(28,6))
    size = 5
    fontsize = 6

    # Distribution of gt_dist vs. pred_dist.
    ax = fig.add_subplot(1,4,1)
    if dist_type == 'prob':
        pred_dist = model.compute_prob_matrix(pred)
    else:
        pred_dist = torch.cdist(pred, pred)
    pred = pred.detach().cpu().numpy()
    gt_dist = gt_dist.detach().cpu().numpy()
    pred_dist = pred_dist.detach().cpu().numpy()
    ax.hist(gt_dist.flatten(), bins=100, label='gt_dist', color='blue')
    ax.hist(pred_dist.flatten(), bins=100, label='pred_dist', color='gray', alpha=0.5)
    ax.legend()
    ax.title.set_text("gt_dist vs. pred_dist")

    # PHATE Embeddings
    ax = fig.add_subplot(1,4,2)
    print(phate_embed.shape)
    phate_embed = phate_embed.detach().cpu().numpy()
    scprep.plot.scatter2d(phate_embed,
                        c=data_clusters,
                        legend=True,
                        ax=ax,
                        title="%s Phate Embedding" % (dataset_name),
                        xticks=True,
                        yticks=True,
                        label_prefix='',
                        fontsize=fontsize,
                        s=size)

    # Predicted Embeddings
    ax = fig.add_subplot(1,4,3)
    title_str = ''
    for metric, value in metrics.items():
        title_str += f"{metric}: {value:.2f}, "
    scprep.plot.scatter2d(pred,
                        c=data_clusters,
                        legend=True,
                        ax=ax,
                        title=title_str,
                        xticks=True,
                        yticks=True,
                        label_prefix='',
                        fontsize=fontsize,
                        s=size)
    
    # Procrustes aligned embeddings
    procrustes_op = Procrustes()
    _, pred_aligned, _ = procrustes_op.fit_transform(phate_embed, 
                                                  pred)
    ax = fig.add_subplot(1,4,4)
    scprep.plot.scatter2d(pred_aligned,
                        c=data_clusters,
                        legend=True,
                        ax=ax,
                        title="Procrustes Aligned",
                        xticks=True,
                        yticks=True,
                        label_prefix='',
                        fontsize=fontsize,
                        s=size)


    plt.tight_layout()
    plt.savefig(save_path)

    print("Saved Embeddings to %s" % (save_path))