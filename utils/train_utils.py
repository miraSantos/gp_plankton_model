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
from evaluation.forecasting_metrics import *

def train_test_split(X, y, train_size):
    """
    :param X:torch.tensor predictor variable
    :param y:torch.tensor dependent variable
    :param train_size:
    :return:
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

    y = y.reshape(len(y))
    # X = X.reshape(len(X))

    if normalize:
        # X = normalize_tensor(X)
        y = normalize_tensor(y)

    X_train, y_train, X_test, y_test = train_test_split(X, y, train_size)
    return X_train, y_train, X_test, y_test


def train_model(likelihood, model, optimizer,config,x_train, y_train,learning_rate=0.1):
    """
    :param likelihood:
    :param model:
    :param learning_rate:
    :return:
    """
    model.train()
    likelihood.train()
    smoke_test = ('CI' in os.environ)
    # Use the adam optimizer
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    pbar = tqdm(range(config["parameters"]["train_iter"]), leave=None)
    for i in pbar:
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        pbar.set_description('Iter %d/%d - Loss: %.3f' % (i + 1, config["parameters"]["train_iter"], loss.item()))
        wandb.log({"Test loss": loss.item()})
        optimizer.step()

def compute_metrics(metrics, actual, predicted ):

    metrics_list = [[] for _ in range(len(metrics))]  # list of lists to store error metric results

    for j in range(len(metrics)):
        metrics_list[j].append(metrics[j](actual,predicted))

    df_metrics = pd.DataFrame({"metrics":metrics,"metrics_values":metrics_list})
    wandb.log({"table":df_metrics})
    return metrics_list


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
    ax.scatter(df.date[:len(x_train)], y_train, color="blue", label="training data")
    ax.scatter(df.date[len(x_train):], y_test, color="red", label="testing data")
    # ax.axvline(x=df.date[len(x_train)], color="red", label="train_test_splot")
    ax.set_title("Dependent: "+ config["dependent"] + " Predictor: "+ config["predictor"] + " " + str(config["parameters"]["train_size"] * 100) + "% of data")
    ax.set_xlabel("Year")
    ax.set_ylabel(config["dependent"])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid()
    plt.show()

    # saving image
    train_test_img = config["res_path"] + "/" + config["dependent"] +"/"+ 'train_test_split_train_size_' + str(
        config["parameters"]["train_size"]) + '.png'
    fig.savefig(train_test_img)
    # wandb.save(train_test_img)
    # wandb.log({"Pre-Training Split": wandb.Image(fig)})

