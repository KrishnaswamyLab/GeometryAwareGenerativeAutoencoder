import torch
import numpy as np
import typing as T
from scipy.special import ive
import pygsp


class HeatKernelKNN:
    """Approximation of the heat kernel. The class has a callable method that computes
    the heat kernel for a given dataset.
    """

    def __init__(self, t: float, knn: int, order: int = 30, graph_type: str = "torch"):
        self.t = t
        self.order = order
        self.knn = knn
        assert graph_type in ["torch", "pygsp"]
        self.graph_type = graph_type

    def get_graph_torch(self, data: torch.Tensor):
        """Get the graph from the dataset.

        Args:
            data (torch.Tensor): dataset of shape (n_samples, n_features)

        Returns:
            torch.Tensor: the graph
        """
        pairwise_dist = torch.cdist(data, data)
        _, indices = torch.topk(pairwise_dist, self.knn, largest=False, dim=-1)
        # Construct the adjacency matrix
        n = data.shape[0]
        A = torch.zeros(n, n, device=data.device)
        for i in range(n):
            A[i, indices[i]] = 1
        A = (A + A.t()) / 2

        # Compute the degree matrix
        degree = A.sum(dim=1)
        inv_deg_sqrt = 1.0 / torch.sqrt(degree)
        D = torch.diag(inv_deg_sqrt)
        L = torch.eye(n, device=data.device) - D @ A @ D

        eigvals = torch.linalg.eigvals(L).real
        max_eigval = eigvals.max()

        return L, max_eigval

    @torch.no_grad()
    def get_graph_pygsp(self, data: torch.Tensor):
        """Get the graph from the dataset.

        Args:
            data (torch.Tensor): dataset of shape (n_samples, n_features)

        Returns:
            pygsp.graphs.Graph: the graph
        """
        graph = pygsp.graphs.NNGraph(data.detach().cpu().numpy(), k=self.knn)
        graph.compute_laplacian("normalized")
        graph.estimate_lmax()
        L = torch.tensor(graph.L.todense()).to(data.device, dtype=data.dtype)
        return L, graph.lmax

    def __call__(self, data: torch.Tensor):
        """Compute the heat kernel for a given dataset.

        Args:
            data (torch.Tensor): dataset of shape (n_samples, n_features)

        Returns:
            [torch.Tensor]: the heat kernel of shape (n_samples, n_samples)
        """
        device = data.device
        if self.graph_type == "torch":
            L, lmax = self.get_graph_torch(data)
        else:
            L, lmax = self.get_graph_pygsp(data)
        cheb_coeff = compute_chebychev_coeff_all(0.5 * lmax, self.t, self.order)
        heat_kernel = expm_multiply(
            L, torch.eye(data.shape[0]).to(device), cheb_coeff, 0.5 * lmax
        )

        # the heat kernel is symmetric, but numerical errors can make it non-symmetric
        heat_kernel = (heat_kernel + heat_kernel.T) / 2
        return torch.tensor(heat_kernel).to(device)


class HeatKernelGaussian:
    """Approximation of the heat kernel with a graph from a gaussian affinity matrix.
    Uses Chebyshev polynomial approximation.
    """

    def __init__(
        self, sigma: float = 1.0, alpha: int = 20, order: int = 30, t: float = 1.0
    ):
        self.sigma = sigma
        self.order = order
        self.t = t
        self.alpha = alpha if alpha % 2 == 0 else alpha + 1

    def __call__(self, data: torch.Tensor):
        L = laplacian_from_data(data, self.sigma, alpha=self.alpha)
        eigvals = torch.linalg.eigvals(L).real
        max_eigval = eigvals.max()
        cheb_coeff = compute_chebychev_coeff_all(0.5 * max_eigval, self.t, self.order)
        heat_kernel = expm_multiply(
            L, torch.eye(data.shape[0]), cheb_coeff, 0.5 * max_eigval
        )
        # symmetrize the heat kernel, for larger t it may not be symmetric
        heat_kernel = (heat_kernel + heat_kernel.T) / 2
        return heat_kernel


def laplacian_from_data(data: torch.Tensor, sigma: float, alpha: int = 20):
    affinity = torch.exp(-torch.cdist(data, data) ** alpha / (2 * sigma**2))
    degree = affinity.sum(dim=1)
    inv_deg_sqrt = 1.0 / torch.sqrt(degree)
    D = torch.diag(inv_deg_sqrt)
    L = torch.eye(data.shape[0]) - D @ affinity @ D
    return L


def expm_multiply(
    L: torch.Tensor,
    X: torch.Tensor,
    coeff: torch.Tensor,
    eigval: T.Union[torch.Tensor, np.ndarray],
):
    """Matrix exponential with chebyshev polynomial approximation."""

    def body(carry, c):
        T0, T1, Y = carry
        T2 = (2.0 / eigval) * torch.matmul(L, T1) - 2.0 * T1 - T0
        Y = Y + c * T2
        return (T1, T2, Y)

    T0 = X
    Y = 0.5 * coeff[0] * T0
    T1 = (1.0 / eigval) * torch.matmul(L, X) - T0
    Y = Y + coeff[1] * T1

    initial_state = (T0, T1, Y)
    for c in coeff[2:]:
        initial_state = body(initial_state, c)

    _, _, Y = initial_state

    return Y


@torch.no_grad()
def compute_chebychev_coeff_all(eigval, t, K):
    return 2.0 * ive(torch.arange(0, K + 1), -t * eigval)
