import torch
import numpy as np
import typing as T
from scipy.special import ive
import pygsp


class HeatKernelCheb:
    """Approximation of the heat kernel. The class has a callable method that computes
    the heat kernel for a given dataset.
    """

    def __init__(self, tau: float, order: int, knn: int):
        self.data_np = data.detach().cpu().numpy()
        self.tau = tau
        self.order = order
        self.knn = knn

    def __call__(self, data: torch.Tensor):
        """Compute the heat kernel for a given dataset.

        Args:
            data (torch.Tensor): dataset of shape (n_samples, n_features)

        Returns:
            [torch.Tensor]: the heat kernel of shape (n_samples, n_samples)
        """
        device = data.device
        graph = pygsp.graphs.NNGraph(data.detach().cpu().numpy(), k=self.knn)
        graph.compute_laplacian("normalized")
        graph.estimate_lmax()
        _filter = pygsp.filters.Heat(graph, self.tau)
        data_np = data.detach().cpu().numpy()
        identity = np.eye(data_np.shape[0])
        heat_kernel = _filter.filter(identity, order=self.order)

        # the heat kernel is symmetric, but numerical errors can make it non-symmetric
        heat_kernel = (heat_kernel + heat_kernel.T) / 2
        return torch.tensor(heat_kernel).to(device)


# TODO: we could use this implementation, but for now it does
# not seem to be necessary.
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


def compute_chebychev_coeff_all(eigval, tau, K):
    return 2.0 * ive(np.arange(0, K + 1), -tau * eigval)


if __name__ == "__main__":
    data = torch.randn(100, 5)
    heat_op = HeatKernelCheb(tau=1.0, order=10, knn=5)
    heat_kernel = heat_op(data)

    # test if symmetric
    assert torch.allclose(heat_kernel, heat_kernel.T)

    # test if positive
    assert torch.all(heat_kernel >= 0)
