import pytest
import torch
from src.heat_kernel import HeatKernelKNN, laplacian_from_data, HeatKernelGaussian


def gt_heat_kernel_knn(
    data,
    t,
    sigma,
):
    L = laplacian_from_data(data, sigma)
    # eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(L)
    # compute the heat kernel
    heat_kernel = eigvecs @ torch.diag(torch.exp(-t * eigvals)) @ eigvecs.T
    return heat_kernel


@pytest.mark.parametrize("graph_type", ["torch", "pygsp"])
def test_heat_kernel_cheb(graph_type):
    data = torch.randn(100, 5)
    heat_op = HeatKernelKNN(t=1.0, order=10, knn=5, graph_type=graph_type)
    heat_kernel = heat_op(data)

    # test if symmetric
    assert torch.allclose(heat_kernel, heat_kernel.T)

    # test if positive
    assert torch.all(heat_kernel >= 0)

    # check if the heat kernel is differentiable
    data = torch.randn(100, 5, requires_grad=True)
    heat_kernel = heat_op(data)
    heat_kernel.sum().backward()
    assert data.grad is not None
    assert torch.all(torch.isfinite(data.grad))


def test_laplacian():
    data = torch.randn(100, 5)
    sigma = 1.0
    L = laplacian_from_data(data, sigma)
    assert torch.allclose(L, L.T)
    # compute the largest eigenvalue
    eigvals = torch.linalg.eigvals(L).real
    max_eigval = eigvals.max()
    min_eigval = eigvals.min()
    assert max_eigval <= 2.0
    torch.testing.assert_allclose(min_eigval, 0.0)


@pytest.mark.parametrize("t", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("order", [10, 30, 50])
def test_heat_kernel_gaussian(t, order):
    data = torch.randn(100, 5)
    heat_op = HeatKernelGaussian(sigma=1.0, t=t, order=order)
    heat_kernel = heat_op(data)

    # test if symmetric
    assert torch.allclose(heat_kernel, heat_kernel.T)

    # test if positive
    assert torch.all(heat_kernel >= 0)

    # test if the heat kernel is close to the ground truth
    gt_heat_kernel = gt_heat_kernel_knn(data, t=t, sigma=1.0)
    assert torch.allclose(heat_kernel, gt_heat_kernel, atol=1e-3)


def test_heat_gauss_differentiable():
    data = torch.randn(100, 5, requires_grad=True)
    heat_op = HeatKernelGaussian(sigma=1.0, t=1.0, order=10)
    heat_kernel = heat_op(data)
    heat_kernel.sum().backward()
    assert data.grad is not None
    assert torch.all(torch.isfinite(data.grad))


if __name__ == "__main__":
    pytest.main([__file__])
