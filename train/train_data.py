import math
import pandas as pd
import numpy as np
import torch
import gpytorch

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

from matplotlib import pyplot as plt
import seaborn as sns

from calendar import month_name as mn
import datetime
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.dates as mdates #for working with dates in plots

# import wandb #library for tracking and visualization

# !git config --global --add safe.directory "/dos/MIT-WHOI/github_repos/syn_model"

def load_data(pkl_path):
    return pd.read_pickle(pkl_path)

def train_test_split(X,y,train_size):
    X_train = X[:math.ceil(train_size*len(X)),]
    y_train = y[:math.ceil(train_size*len(X)),]
    X_test = X[math.ceil(train_size*len(y)):,]
    y_test = y[math.ceil(train_size*len(y)):,]
    return X_train, y_train, X_test, y_test

def define_training_data(df,train_size):
    X = torch.tensor(df[["lindex"]].values, dtype=torch.float64)
    y = torch.tensor(df[["log_syn_interpolated"]].values)
    y = y.reshape(len(y))
    X = X.reshape(len(X))

    X_train, y_train, X_test, y_test = train_test_split(X, y, train_size)
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_train dtype: ", X_train.dtype)
    print("y_train dtpye: ", y_train.dtype)
    return X_train, y_train, X_test, y_test

class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x,train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4)
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


if __name__ == '__main__':
    df = load_data(pkl_path="/home/mira/PycharmProjects/gp_plankton_model/datasets/syn_dataset.pkl")
    print(df.columns)
    X_train, y_train, X_test, y_test = define_training_data(df, train_size=1/50)
    print(X_train.shape)
    print(y_train.shape)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SpectralMixtureGPModel(X_train, y_train, likelihood)

    import os

    smoke_test = ('CI' in os.environ)
    training_iter = 2000 if smoke_test else 100

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print("TRAINING FINISHED")
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Make predictions
        observed_pred = likelihood(model(X_test))

        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(X_train.numpy(), y_train.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(X_test.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(X_test.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        # ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.show()
