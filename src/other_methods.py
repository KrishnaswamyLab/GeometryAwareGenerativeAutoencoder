'''
Adapted from https://github.com/KrishnaswamyLab/HeatGeo/blob/2274f1048d5dd41ade56063e74be7b3403a26894/heatgeo/other_emb.py
'''

import numpy as np
import pygsp
import graphtools
from graphtools.matrix import set_diagonal, to_array
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform

def get_alpha_decay_graph(
    X,
    knn: int = 5,
    decay: float = 40.0,
    anisotropy: float = 0,
    n_pca: int = None,
    **kwargs
):
    return graphtools.Graph(
        X,
        knn=knn,
        decay=decay,
        anisotropy=anisotropy,
        n_pca=n_pca,
        use_pygsp=True,
        random_state=42,
    ).to_pygsp()

def diff_op(graph):
    """
    Compute the diffusion operator for a pygsp graph.
    """
    assert isinstance(graph, pygsp.graphs.Graph)
    K = set_diagonal(graph.W, 1)
    diff_op_ = normalize(K, norm="l1", axis=1)
    return diff_op_


class DiffusionMap():
    """Diffusion Map embedding with different graph construction."""

    def __init__(
        self,
        knn: int = 5,
        decay: int = 10,
        n_pca: int = 40,
        tau: float = 1,
        emb_dim: int = 2,
        anisotropy: int = 0,
        graph_type: str = "alpha",
        **kwargs
    ):
        self.knn = knn
        self.decay = decay
        self.n_pca = n_pca
        self.tau = tau
        self.emb_dim = emb_dim
        self.anisotropy = anisotropy
        self.graph_type = graph_type
        self.kwargs = kwargs


    def fit(self, data):
        self.graph = get_alpha_decay_graph(
                data,
                knn=self.knn,
                decay=self.decay,
                anisotropy=self.anisotropy,
                n_pca=self.n_pca,
            )
        self.graph.compute_laplacian(lap_type="normalized")


    def metric_computation(self, data):
        # P = self.graph.P.toarray()
        P = diff_op(self.graph).toarray()
        eval, evec = np.linalg.eig(P)
        eval, evec = np.real(eval), np.real(evec)
        eval = eval**self.tau
        emb = eval[None, :] * evec
        self.dist = squareform(pdist(emb))

    def diffusion_emb(self, data):
        # P = self.graph.P.toarray()
        P = diff_op(self.graph).toarray()
        eval, evec = np.linalg.eig(P)
        eval = eval**self.tau
        emb = eval[None, :] * evec
        return emb

    def fit_transform(
        self,
        data,
    ) -> np.ndarray:
        self.fit(data)
        P = diff_op(self.graph).toarray()
        eval, evec = np.linalg.eig(P)
        eval, evec = np.real(eval), np.real(evec)
        eval = eval**self.tau
        order_eval = np.argsort(np.abs(eval))[::-1]
        self.emb = (
            eval[None, order_eval[: self.emb_dim]] * evec[:, order_eval[: self.emb_dim]]
        )
        return self.emb

if __name__ == "__main__":
    data = np.random.rand(100, 3) # 100 samples, 3 features
    dm = DiffusionMap()
    emb = dm.fit_transform(data, )
    print(emb.shape) # (100, 2)