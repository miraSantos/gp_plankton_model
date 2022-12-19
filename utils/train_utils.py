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
import matplotlib.pyplot as plt

import wandb  # library for tracking and visualization
from evaluation.forecasting_metrics import *

wandb

def train_test_split(X, y, train_size):
    """
    :param X:torch.tensor predictor variable
    :param y:torch.tensor dependent variable
    :param train_size:float32 size of training data expressed as decimal between 0 and 1.
    :return X_train:
    """
    X_train = X[:math.ceil(train_size * len(X)), ]
    y_train = y[:math.ceil(train_size * len(y)), ]
    X_test = X[math.ceil(train_size * len(X)):, ]
    y_test = y[math.ceil(train_size * len(y)):, ]
    return X_train, y_train, X_test, y_test


def normalize_tensor(tensor):
    return (tensor - torch.nanmean(tensor)) / np.nanstd(tensor.numpy())

def define_training_data(X, y, train_size, normalize=True):
    """
    :param X:
    :param y:
    :param train_size:
    :param normalize:
    :return:
    """

    # y = y.reshape(len(y))
    # X = X.reshape(len(X))

    if normalize:
        X = normalize_tensor(X)
        y = normalize_tensor(y)

    X_train, y_train, X_test, y_test = train_test_split(X, y, train_size)
    return X_train, y_train, X_test, y_test


def train_model(likelihood, model, optimizer,x_train, y_train):
    """
    :param likelihood:
    :param model:
    :return:
    """
    model.train()
    likelihood.train()
    smoke_test = ('CI' in os.environ)
    # Use the adam optimizer
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    pbar = tqdm(range(wandb.config.train_iter), leave=None)
    for i in pbar:
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        pbar.set_description('Iter %d/%d - Loss: %.3f  noise: %.3f' % (i + 1,
                                                                 wandb.config.train_iter,
                                                                       loss.item(),
                                                                   model.likelihood.noise.item()
                                                                                            ))
        wandb.log({"Test loss": loss.item(),
                    "Noise" : model.likelihood.noise.item()})
        optimizer.step()

def plot_train_test_data(df,x_train, y_train, x_test, y_test,config):
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
    ax.scatter(df.date[:len(x_train)], y_train, c="blue", marker=".", label="training data")
    ax.scatter(df.date[len(x_train):], y_test, c="red", marker="." , label="testing data")
    # ax.axvline(x=df.date[len(x_train)], color="red", label="train_test_splot")
    ax.set_title("Dependent: "+ config["dependent"] + " Predictor: "+ config["predictor"] + " " +
                 str(wandb.config.train_size * 100) + "% of data")
    ax.set_xlabel("Year")
    ax.set_ylabel(config["dependent"])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid()
    ax.legend()
    plt.show()

    # saving image
    results_folder = config["res_path"] + "/" + config["dependent"] + "/"
    # Check whether the specified path exists or not
    isExist = os.path.exists(results_folder)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(results_folder)
        print("The new directory is created!")

    train_test_img = config["res_path"] + "/" + config["dependent"] + "/" + 'train_test_split_train_size_' + str(
        wandb.config.train_size) + '.png'
    fig.savefig(train_test_img)
    wandb.save(train_test_img)
    wandb.log({"Pre-Training Split": train_test_img})
    plt.close(fig)

def load_test_train(config):
    df = pd.read_csv(config["data_path"], low_memory=False)
    df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'],
                                       format="%Y-%m-%d")  # required or else dates start at 1971! (WEIRD BUG HERE)
    dfsubset = df.dropna(subset=config[
        "dependent"])  # dropping na values #TODO: fix spectral model so that it can handle missing observations

    if wandb.config.num_dims > 1:
        dfsubset = df.dropna(subset=[config["dependent"], config[
            "predictor"]])  # dropping na values #TODO: fix spectral model so that it can handle missing observations
        X = torch.tensor(dfsubset.loc[:, config["predictor"]].reset_index().to_numpy(),
                         dtype=torch.float32)  # 2D tensor
    elif wandb.config.num_dims == 1:
        X = torch.tensor(dfsubset.index, dtype=torch.float32)
    else:
        assert wandb.config.num_dims > 0, f"number greater than 0 expected, got: {wandb.config.num_dims}"

    if config["take_log"]:
        dependent = np.log(dfsubset[config["dependent"]].values)
    else:
        dependent = dfsubset[config["dependent"]].values
    y = torch.tensor(dependent, dtype=torch.float32)

    # #defining training data based on testing split
    X_train, y_train, X_test, y_test = define_training_data(X, y, train_size=wandb.config.train_size,
                                                            normalize=config["normalize"])

    wandb.config.X_train_shape = X_train.shape
    wandb.config.y_train_shape = y_train.shape
    wandb.config.X_test_shape = X_test.shape
    wandb.config.y_test_shape = y_test.shape

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)

    return dfsubset, X, X_train, y_train, X_test, y_test


