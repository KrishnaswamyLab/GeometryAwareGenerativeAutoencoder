import numpy as np
import graphtools

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

class DiffusionMap():
    def __init__(self, n_components=2, t=3, **kwargs) -> None:
        self.n_components = n_components
        self.kwargs = kwargs
        self.t = t
    def fit(self, data):
        G = get_alpha_decay_graph(data, **self.kwargs)
        Dinvhf = np.diag(1 / np.sqrt(G.K.toarray().sum(axis=1)))
        self.Dinvhf = Dinvhf
        A = Dinvhf @ G.K @ Dinvhf
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.emb_dm = Dinvhf @ eigenvectors[:, :self.n_components] * eigenvalues[:self.n_components] ** self.t
        return self
    def transform(self, data):
        return self.emb_dm
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)