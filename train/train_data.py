import math
import pandas as pd
import numpy as np
import torch
import gpytorch
import os
from tqdm import tqdm
import random

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

from matplotlib import pyplot as plt

import wandb #library for tracking and visualization

def load_data(pkl_path):
    return pd.read_pickle(pkl_path)


def train_test_split(X, y, train_size):
    """
    :param X:
    :param y:
    :param train_size:
    :return:
    """
    X_train = X[:math.ceil(train_size * len(X)), ]
    y_train = y[:math.ceil(train_size * len(y)), ]
    X_test = X[math.ceil(train_size * len(X)):, ]
    y_test = y[math.ceil(train_size * len(y)):, ]
    return X_train, y_train, X_test, y_test


def normalize_tensor(tensor):
    return (tensor - torch.mean(tensor)) / torch.std(tensor)

def define_training_data(X, y, train_size, normalize=True):
    """
    :param X:
    :param y:
    :param train_size:
    :param normalize:
    :return:
    """

    y = y.reshape(len(y))
    # X = X.reshape(len(X))

    if normalize:
        # X = normalize_tensor(X)
        y = normalize_tensor(y)

    X_train, y_train, X_test, y_test = train_test_split(X, y, train_size)
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_train dtype: ", X_train.dtype)
    print("y_train dtpye: ", y_train.dtype)

    return X_train, y_train, X_test, y_test


class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_mixtures):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures)
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_model(likelihood, model, learning_rate=0.1):
    """
    :param likelihood:
    :param model:
    :param learning_rate:
    :return:
    """
    model.train()
    likelihood.train()
    smoke_test = ('CI' in os.environ)
    training_iter = 2000 if smoke_test else 100
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    pbar = tqdm(range(training_iter))
    for i in pbar:
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        pbar.set_description('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()


def evaluate_model(X_test, likelihood, model):
    """
    :param X_test:
    :param likelihood:
    :param model:
    :return:
    """
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Make predictions
        observed_pred = likelihood(model(X_train))

        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(X_train, y_train, 'k*')
        # Plot predictive means as blue line
        ax.plot(X_train, observed_pred.mean, 'b')
        ax.plot(X_test, y_test, 'g')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(X_test, lower, upper, alpha=0.5)
        # ax.set_ylim([-.025, .025])
        ax.legend(['Observe Data', 'Mean', 'Confidence'])
        plt.show()


if __name__ == '__main__':
    df = load_data(pkl_path="/home/mira/PycharmProjects/gp_plankton_model/datasets/syn_dataset.pkl")
    train_size = 1/2

    X = torch.tensor(df['lindex'].values, dtype=torch.float32)
    y = torch.tensor(df["log_syn_interpolated"].values, dtype=torch.float32)

    X_train, y_train, X_test, y_test = define_training_data(X,y, train_size=train_size, normalize=True)
    print(X_train.shape)
    print(y_train.shape)
    plt.scatter(X_train, y_train)
    plt.scatter(X_test, y_test)
    plt.title("Normalized X_train vs. y_train")
    plt.show()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-3))
    num_mixtures = 4
    model = SpectralMixtureGPModel(X_train, y_train, likelihood, num_mixtures)
    learning_rate = 0.1

    train_model(likelihood, model, learning_rate=learning_rate)
    print("TRAINING FINISHED")
    evaluate_model(X_test, likelihood, model)
    # print("EVALUATION FINISHED")
    # df = df.asfreq("W") #setting frequency as week


