import math
import pandas as pd
import torch
import gpytorch
import os
from tqdm import tqdm
from models import exactGP_model

from matplotlib.dates import YearLocator

from PIL import Image

from matplotlib import pyplot as plt

import wandb  # library for tracking and visualization


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


def train_model(likelihood, model, learning_rate=0.1):
    """
    :param likelihood:likelihood
    :param model:class
    :param learning_rate:float
    :return:
    """
    model.train()
    likelihood.train()

    smoke_test = ('CI' in os.environ)
    config.training_iter = 2000 if smoke_test else 100

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    pbar = tqdm(range(config.training_iter))
    for i in pbar:
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        pbar.set_description('Iter %d/%d - Loss: %.3f' % (i + 1, config.training_iter, loss.item()))
        wandb.log({"Test loss": loss.item()})
        optimizer.step()


def plot_train_test_data(x_train, y_train, x_test, y_test):
    # wandb.log({"train_test_data": wandb.plot.line_series(
    #     xs=[x_train, x_test],
    #     ys=[y_train, y_test],
    #     keys=["train", "test"],
    #     title="Training and testing sets",
    #     xname="Number of Observations"
    # )})
    width = 20
    height = 5
    fig, ax = plt.subplots(figsize=(width, height))
    ax.scatter(df.date[len(X_train):], y_test, color="blue")
    ax.scatter(df.date[:len(X_train)], y_train, color="red")
    ax.axvline(x=df.date[len(X_train)], color="red", label="train_test_splot")
    ax.set_title("Training and Testing Split")
    ax.set_xlabel("Time")
    ax.set_ylabel("[Syn]")
    ax.xaxis.set_major_locator(YearLocator(base=1))
    ax.grid()
    train_test_img = res_path + 'training_testing_split.png'
    fig.savefig(train_test_img)
    im = Image.open(train_test_img)
    wandb.log({"Pre-Training Split": wandb.Image(im)})


if __name__ == '__main__':
    wandb.login()
    res_path = "/home/mira/PycharmProjects/gp_plankton_model/results/"
    PATH = "/home/mira/PycharmProjects/gp_plankton_model/datasets/syn_dataset.pkl"
    df = load_data(pkl_path=PATH)

    wandb.init(project="syn_model")
    config = wandb.config
    config.data_path = PATH
    config.train_size = 1 / 2
    config.num_mixtures = 4
    config.learning_rate = 0.1
    config.predictor = 'lindex'
    config.dependent = 'log_syn_interpolated'

    X = torch.tensor(df[config.predictor].values, dtype=torch.float32)
    y = torch.tensor(df[config.dependent].values, dtype=torch.float32)

    X_train, y_train, X_test, y_test = define_training_data(X, y, train_size=config.train_size, normalize=True)

    config.X_train_shape = X_train.shape
    config.y_train_shape = y_train.shape
    config.X_test_shape = X_test.shape
    config.y_test_shape = y_test.shape

    plot_train_test_data(X_train, y_train, X_test, y_test)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-3))

    model = spectral_model.SpectralMixtureGPModel(X_train, y_train, likelihood, config.num_mixtures)
    config.model_type = "spectralGP"

    wandb.watch(model, log="all")

    train_model(likelihood, model, learning_rate=config.learning_rate)

    # saving model checkpoint
    torch.save(model.state_dict(),
               res_path + config.model_type + "_training_iter_" + str(config.training_iter) + "_model.h5")
    wandb.save(res_path + "training_iter_" + str(config.training_iter) + "_model.h5")

    evaluate_model(X_test, likelihood, model)
