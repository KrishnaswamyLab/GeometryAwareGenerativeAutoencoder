import torch
import numpy as np
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.preprocessing import PowerTransformer as PT

class LogTransform():
    def __init__(self, eps=1e-10, device=None):
        self.eps = eps
    def transform(self, X):
        return torch.log(X+self.eps)
    def fit_transform(self, X):
        return self.transform_cpu(X)
    def transform_cpu(self, X):
        return np.log(X+self.eps)
    
class NonTransform():
    def __init__(self, device=None):
        pass
    def transform(self, X):
        return X
    def fit_transform(self, X):
        return X
    
class StandardScaler():
    def __init__(self):
        self.ss = SS()
        self.mean_ = None
        self.std_ = None
    def fit_transform(self, X):
        res = self.ss.fit_transform(X)
        self.mean_ = torch.tensor(self.ss.mean_)
        self.std_ = torch.tensor(self.ss.scale_)
        return res
    def transform(self, X):
        return standard_scale_transform_torch(X, self.mean_, self.std_)

class MinMaxScaler():
    def __init__(self):
        self.mms = MMS()
        self.min_ = None
        self.scale_ = None
    def fit_transform(self, X):
        res = self.mms.fit_transform(X)
        self.min_ = torch.tensor(self.mms.min_)
        self.scale_ = torch.tensor(self.mms.scale_)
        return res
    def transform(self, X):
        return minmax_scale_transform_torch(X, self.min_, self.scale_)

class PowerTransformer():
    def __init__(self):
        self.pt = PT()
        self.lambdas_ = None
    def fit_transform(self, X):
        res = self.pt.fit_transform(X)
        self.lambdas_ = torch.tensor(self.pt.lambdas_)
        return res
    def transform(self, X):
        return standard_scale_transform_torch(
            yeo_johnson_transform_torch(X, self.lambdas_),
            torch.tensor(self.pt._scaler.mean_),
            torch.tensor(self.pt._scaler.scale_)
        )
    
class KernelTransform():
    def __init__(self, type, sigma=1, epsilon=1, alpha=10, use_std=True):
        assert type in ['gaussian', 'alpha_decaying']
        self.sigma = sigma
        self.epsilon = epsilon
        self.alpha = alpha
        self.use_std = use_std
        if use_std:
            self.std = 1
        if type == 'gaussian':
            self.kernel = lambda x: gaussian_kernel(x, sigma=sigma)
        elif type == 'alpha_decaying':
            self.kernel = lambda x: alpha_decaying_kernel(x, epsilon=epsilon, alpha=alpha)

    def fit_transform(self, X):
        X = torch.tensor(X)
        if self.use_std:
            self.std = torch.std(X)
            X = X / self.std
        tx = self.kernel(X)
        return tx.cpu().numpy()
    
    def transform(self, X):
        if self.use_std:
            X = X / self.std
        return self.kernel(X)

def gaussian_kernel(x, sigma=1):
    """
    Computes the Gaussian kernel between two PyTorch tensors.
    
    Parameters:
    X (torch.Tensor): The first tensor.
    Y (torch.Tensor): The second tensor.
    sigma (float): The kernel bandwidth.
    
    Returns:
    torch.Tensor: The kernel matrix.
    """
    return torch.exp(-torch.pow(x, 2) / (2 * sigma ** 2))

def alpha_decaying_kernel(x, epsilon, alpha=10):
    r"""Calculate the $\alpha$-decaying kernel function.

    Calculate a simplified variant of the $\alpha$-decaying kernel as
    described by Moon et al. [1]. In contrast to the original
    description, this kernel only uses a *single* local density estimate
    instead of per-point estimates.

    Parameters
    ----------
    X : np.array of size (n, n)
        Input data set.

    epsilon : float
        Standard deviation or local scale parameter. This parameter is
        globally used and does *not* depend on the local neighbourhood
        of a point.

    alpha : float
        Value for the decay.

    Returns
    -------
    Kernel matrix.

    References
    -----
    [1]: Moon et al., Visualizing Structure and Transitions for
    Biological Data Exploration, Nature Biotechnology 37, pp. 1482–1492,
    2019. URL: https://www.nature.com/articles/s41587-019-0336-3
    """
    return torch.exp(-(x / epsilon)**alpha)


def standard_scale_transform_torch(X, mean_, std_):
    return (X - mean_.to(device=X.device, dtype=X.dtype)) / std_.to(device=X.device, dtype=X.dtype)

def minmax_scale_transform_torch(X, min_, scale_):
    return (X - min_.to(device=X.device, dtype=X.dtype)) * scale_.to(device=X.device, dtype=X.dtype)


def yeo_johnson_transform_torch(X, lambdas):
    """
    Applies the Yeo-Johnson transformation to a PyTorch tensor.
    
    Parameters:
    X (torch.Tensor): The data to be transformed.
    lambdas (torch.Tensor or ndarray): The lambda parameters from the fitted sklearn PowerTransformer.
    
    Returns:
    torch.Tensor: The transformed data.
    """
    lambdas = lambdas.to(device=X.device, dtype=X.dtype)
    X_transformed = torch.zeros_like(X, device=X.device, dtype=X.dtype)
    
    # Define two masks for the conditional operation
    positive = X >= 0
    negative = X < 0

    # Applying the Yeo-Johnson transformation
    # For positive values
    pos_transform = torch.where(
        lambdas != 0,
        torch.pow(X[positive] + 1, lambdas) - 1,
        torch.log(X[positive] + 1)
    ) / lambdas

    # For negative values (only if lambda != 2)
    neg_transform = torch.where(
        lambdas != 2,
        -(torch.pow(-X[negative] + 1, 2 - lambdas) - 1) / (2 - lambdas),
        -torch.log(-X[negative] + 1)
    )

    # Assigning the transformed values back to the tensor
    X_transformed[positive] = pos_transform
    X_transformed[negative] = neg_transform

    return X_transformed
