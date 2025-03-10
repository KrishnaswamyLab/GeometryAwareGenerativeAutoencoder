import csv
import scipy.io as spio
import numpy as np
import math
import pandas as pd

import torch
import gpytorch
from gpytorch.kernels import RBFKernel, MaternKernel, PolynomialKernel, LinearKernel, ScaleKernel, AdditiveKernel

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('./EB_x_y.csv')

inputs_train = np.array(data)[:,50:]

# Define a simple GP model
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# No training data is needed for the GP prior
x = torch.zeros([1,3])  # Dummy values
y = torch.zeros(1)  # Dummy values
likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood.noise = torch.tensor(0.05)  # Sets the noise level to 0.1

# Initialize the GP model
model = GPModel(x, y, likelihood)

lengthscale = 4.0
model.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale)

###

# Put the model in evaluation mode since we're working with a prior (no training needed)
model.eval()

# Get the GP prior distribution over inputs_test
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Get the MultivariateNormal prior
    prior_dist = model(torch.tensor(inputs_train).float())
    
    # Sample from the prior distribution
    gp_samples = prior_dist.sample()

x_train = inputs_train[:, 0]
y_train = inputs_train[:, 1]
z_train = inputs_train[:, 2]
preds_train = gp_samples.numpy().flatten()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x_train, y_train, z_train, c=preds_train, cmap='viridis', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('X')

cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('GP Sample Values')

plt.savefig('gp_samples_plot.pdf', format='pdf')

###

labels_train = preds_train

num_points_list = [1000, 2000, 3000, 4000, 5000, 10000, 100000]

for num_points in num_points_list:
    min_val = -3.5
    max_val = 3.5
    inputs_test = np.random.uniform(low=min_val, high=max_val, size=(num_points, 3))

    labels_train = np.ones(len(inputs_train))
    # Instantiate the model with training data
    model = GPModel(torch.tensor(inputs_train).float(), torch.tensor(labels_train).float(), likelihood)

    # Switch to evaluation mode for posterior computation
    model.eval()
    likelihood.eval()

    # Compute the GP posterior for test points inputs_test, conditioned on inputs_train and labels_train
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model(torch.tensor(inputs_test).float())  # This computes the posterior conditioned on inputs_train and labels_train

    # Extract the posterior variance (uncertainty) at the points in inputs_test
    posterior_variance_test = posterior.variance

    x_test = inputs_test[:, 0]
    y_test = inputs_test[:, 1]
    z_test = inputs_test[:, 2]
    preds_test = posterior_variance_test.detach().numpy().flatten()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(x_test, y_test, z_test, c=preds_test, cmap='viridis', marker='o')

    # Compute the GP posterior for inputs_train, conditioned on inputs_train and labels_train
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model(torch.tensor(inputs_train).float())  # This computes the posterior conditioned on inputs_train and labels_train

    # Extract the posterior variance (uncertainty) at the points in inputs_test
    posterior_variance_train = posterior.variance
    print('inputs_train', inputs_train.shape)
    print('posterior_variance_train', posterior_variance_train.shape)
    print('inputs_test', inputs_test.shape)
    print('posterior_variance_test', posterior_variance_test.shape)


    x_train = inputs_train[:, 0]
    y_train = inputs_train[:, 1]
    z_train = inputs_train[:, 2]
    preds_train = posterior_variance_train.numpy().flatten()

    scatter = ax.scatter(x_train, y_train, z_train, c=preds_train, cmap='viridis', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('OOD Indicator')

    plt.savefig(f'{num_points}_samples.pdf', format='pdf')

### Create envelope.

num_points_list = [5000, 10000, 30000, 60000, 100000]
for num_points in num_points_list:

    # Define the interval [min_var, max_var]
    min_var = 0.1
    max_var = 0.3

    inputs_test_eval = inputs_test[:num_points]
    preds_test_eval = preds_test[:num_points]

    # Create a mask for values in preds_test that are within the interval [min_var, max_var]
    mask = (preds_test_eval >= min_var) & (preds_test_eval <= max_var)

    # Apply the mask to filter preds_test and inputs_test
    filtered_inputs_test = inputs_test_eval[mask, :]  # Filter rows of inputs_test based on the mask
    filtered_preds_test = preds_test_eval[mask]  # Filter preds_test by the mask

    x_subset = filtered_inputs_test[:, 0]
    y_subset = filtered_inputs_test[:, 1]
    z_subset = filtered_inputs_test[:, 2]
    preds_subset = filtered_preds_test

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # scatter_test = ax.scatter(x_subset, y_subset, z_subset, c='blue', marker='o')
    scatter_test = ax.scatter(x_subset, y_subset, z_subset, c=preds_subset, cmap='viridis', marker='o')

    # Compute the GP posterior for inputs_train, conditioned on inputs_train and labels_train
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model(torch.tensor(inputs_train).float())  # This computes the posterior conditioned on inputs_train and labels_train

    # Extract the posterior variance (uncertainty) at the points in inputs_train
    posterior_variance_train = posterior.variance

    x_train = inputs_train[:, 0]
    y_train = inputs_train[:, 1]
    z_train = inputs_train[:, 2]
    preds_train = posterior_variance_train.numpy().flatten()

    scatter_train = ax.scatter(x_train, y_train, z_train, c='red', marker='o')
    # scatter = ax.scatter(x_subset, y_subset, z_subset, c=preds_train, cmap='viridis', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')

    cbar = fig.colorbar(scatter_test, ax=ax)
    cbar.set_label('OOD Indicator')

    plt.savefig(f'envelope_{num_points}_samples.pdf', format='pdf')

###

print("Done.")