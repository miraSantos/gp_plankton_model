import os
import sys


sys.path.append(os.getcwd())

import gpytorch
import matplotlib.dates as mdates  # v 3.3.2
import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
import yaml
from PIL import Image

from evaluation.forecasting_metrics import *


import models.spectralGP_model
import argparse

def plot_inference(df,X_test, y_test, X_train, y_train):
    """
    :param x_test:
    :param likelihood:
    :param model:
    :return:
    """
    # Initialize plot

    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    width = 20
    height = 5
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training preprocess as black stars
    ax.plot(df.date[:len(X_train)], y_train, 'k*', label="training data")
    # Plot predictive means as blue line
    ax.plot(df.date, observed_pred.mean.detach().numpy(), 'b', label="prediction")
    #plot testing data
    ax.plot(df.date[len(X_train):], y_test, 'g', label="testing data")
    # Shade between the lower and upper confidence bounds
    ax.fill_between(df.date, lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Year")
    ax.set_ylabel("[log(Syn)] (Normalized")
    ax.legend()
    ax.grid()
    eval_img = train_config["res_path"] +"/eval_train_size_" + str(config.train_size) + '.png'
    ax.set_title("Evaluation " + "Training Size " + str(config.train_size*100) + "% of data")
    fig.savefig(eval_img)
    im = Image.open(eval_img)
    wandb.log({"Evaluation": wandb.Image(im)})

def compute_metrics(metrics, actual, predicted ):

    metrics_list = [[] for _ in range(len(metrics))]  # list of lists to store error metric results

    for j in range(len(metrics)):
        metrics_list[j].append(metrics[j](actual,predicted))

    metrics_dict = {metrics[i].__name__: metrics_list[i] for i in range(len(metrics))}

    wandb.log(metrics_dict)
    return metrics_list

if __name__ == '__main__':
    with open("train/spectral_model_config.yaml", "r") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    slurm_id = sys.argv[1]
    print("slurm_id")

    wandb.login()

    wandb.init(project="syn_model_evaluation")
    config = wandb.config
    config.train_size = int(slurm_id) /10
    config.num_mixtures = train_config["mixtures"]
    config.learning_rate = train_config["learning_rate"]
    config.predictor = 'daily_index'
    config.dependent = train_config["dependent"]
    config.num_dims = train_config["num_dims"]
    config.noise_prior_loc = train_config["noise_prior_loc"]
    config.noise_prior_scale = train_config["noise_prior_scale"]

    likelihood = gpytorch.likelihoods.GaussianLikelihoodWithMissingObs(noise_prior=gpytorch.priors.NormalPrior(config.noise_prior_loc, config.noise_prior_scale))

    df = pd.read_csv(train_config["data_path"])
    dfsubset = df.dropna(subset=config.dependent)

    X = torch.load(train_config["split_folder"] + "X_dataset.pt")
    X_train = torch.load(train_config["split_folder"] + "train_size_" + str(train_config["train_size"]) + "_X_train.pt")
    y_train = torch.load(train_config["split_folder"] + "train_size_" + str(train_config["train_size"]) + "_y_train.pt")
    X_test = torch.load(train_config["split_folder"] + "train_size_" + str(train_config["train_size"]) + "_X_test.pt")
    y_test = torch.load(train_config["split_folder"] + "train_size_" + str(train_config["train_size"]) + "_y_test.pt")

    model = models.spectralGP_model.SpectralMixtureGPModel(X_train, y_train, likelihood, train_config["mixtures"],
                                                           config.num_dims)

    model.load_state_dict(torch.load(train_config["model_checkpoint_folder"] + "/spectral_model_training_size_" +
                                     str(config.train_size) + "_model_checkpoint.pt"))

    model.eval()

    # generrate predictions
    observed_pred = likelihood(model(torch.tensor(dfsubset.index, dtype=torch.float32)))
    print(observed_pred)

    # plot inference
    plot_inference(dfsubset, X_test, y_test, X_train, y_train)

    metrics = [me, rae, mape, rmse, mda]  # list of metrics to compute see forecasting_metrics.p
    actual = y_test.numpy()
    predicted = observed_pred[len(X_train):].mean.detach().numpy()
    print(len(actual))
    print(len(predicted))

    result = compute_metrics(metrics, actual, predicted)
    print(result)
