import math
import pandas as pd
import torch
import gpytorch
import os
import sys
import numpy as np
import matplotlib.dates as mdates  # v 3.3.2

sys.path.append(os.getcwd())

from tqdm import tqdm
import yaml
import models.spectralGP_model
import matplotlib.pyplot as plt
import argparse

from matplotlib.dates import YearLocator
from PIL import Image
import wandb  # library for tracking and visualization

wandb.login()


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


def train_model(likelihood, model, optimizer, learning_rate=0.1):
    """
    :param likelihood:
    :param model:
    :param learning_rate:
    :return:
    """
    model.train()
    likelihood.train()
    smoke_test = ('CI' in os.environ)
    config.training_iter = train_config["training_iter"]
    # Use the adam optimizer
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    pbar = tqdm(range(config.training_iter), leave=None)
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

    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")

    width = 20
    height = 5
    fig, ax = plt.subplots(figsize=(width, height))
    ax.scatter(df.date[:len(X_train)], y_train, color="blue", label="training data")
    ax.scatter(df.date[len(X_train):], y_test, color="red", label="testing data")
    ax.axvline(x=df.date[len(X_train)], color="red", label="train_test_splot")
    ax.set_title("Train Test Split " + "Training Size " + str(config.train_size * 100) + "% of data")
    ax.set_xlabel("Year")
    ax.set_ylabel("[log(Syn)] (Normalized")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.grid()

    # saving image
    train_test_img = train_config["res_path"] + 'train_test_split_train_size_' + str(
        config.train_size) + '.png'
    fig.savefig(train_test_img)
    im = Image.open(train_test_img)
    wandb.log({"Pre-Training Split": wandb.Image(im)})


if __name__ == '__main__':
    with open("train/train_config_local.yaml", "r") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    wandb.login()

    wandb.init(project="syn_model")
    config = wandb.config
    config.train_size = train_config["train_size"]  # passing thru slurm id to parallelize train size
    config.num_mixtures = train_config["num_mixtures"]
    config.learning_rate = train_config["learning_rate"]
    config.predictor = 'daily_index'
        config.dependent = train_config["dependent"]

    df = pd.read_csv(train_config["data_path"])

    X = torch.tensor(df.index, dtype=torch.float32)
    y = torch.tensor(df[config.dependent].values, dtype=torch.float32)

    X_train, y_train, X_test, y_test = define_training_data(X, y, train_size=config.train_size, normalize=True)

    torch.save(X, train_config["split_folder"] + "X_dataset.pt")
    torch.save(X_train, train_config["split_folder"] + "train_size_" + str(config.train_size) + "_X_train.pt")
    torch.save(y_train, train_config["split_folder"] + "train_size_" + str(config.train_size) + "_y_train.pt")
    torch.save(X_test, train_config["split_folder"] + "train_size_" + str(config.train_size) + "_X_test.pt")
    torch.save(y_test, train_config["split_folder"] + "train_size_" + str(config.train_size) + "_y_test.pt")

    config.X_train_shape = X_train.shape
    config.y_train_shape = y_train.shape
    config.X_test_shape = X_test.shape
    config.y_test_shape = y_test.shape
    config.sp_mixture_better_lower_bound = 1e-3

    plot_train_test_data(X_train, y_train, X_test, y_test)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = models.spectralGP_model.SpectralMixtureGPModel(X_train, y_train, likelihood, config.num_mixtures)

    wandb.watch(model, log="all")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_model(likelihood, model, optimizer, learning_rate=config.learning_rate)

    # saving model checkpoint
    torch.save(model.state_dict(), train_config["model_checkpoint_folder"] + "/training_size_" +
               str(config.train_size) + "_model_checkpoint.pt")
