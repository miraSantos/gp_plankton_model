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
    fig, ax = plt.subplots(1, 2, figsize=(width, height))
    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training preprocess as black stars
    ax[0].plot(df.date[:len(X_train)], y_train, 'k*', label="training data")
    # Plot predictive means as blue line
    ax[0].plot(df.date, observed_pred.mean.detach().numpy(), 'b', label="prediction")
    #plot testing data
    ax[0].plot(df.date[len(X_train):], y_test, 'g', label="testing data")
    # Shade between the lower and upper confidence bounds
    ax[0].fill_between(df.date, lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
    ax[0].xaxis.set_major_locator(mdates.YearLocator())
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax[0].set_xlabel("Year")
    ax[0].set_ylabel("Syn Concentration")
    ax[0].legend()
    ax[0].grid()
    ax[0].set_title("Evaluation " + "Training Size " + str(train_config["train_size"]*100) + "% of data")


    #plotting with DOY on the x-axis
    ax[1].scatter(df.doy_numeric[len(X_train):], y_test, label="observations",c = "green")
    ax[1].scatter(df.doy_numeric[len(X_train):], observed_pred.mean.detach().numpy()[len(X_train):],label = "prediction",c = "blue")
    ax[1].set_xlabel("Day of the Year")
    ax[1].set_ylabel("Syn Conc")
    ax[1].legend()
    eval_img = train_config["res_path"] +"/eval_train_size_" + str(train_config["train_size"]) + '.png'
    fig.savefig(eval_img)
    im = Image.open(eval_img)
    wandb.log({"Evaluation": wandb.Image(im)})



def compute_metrics(metrics, actual, predicted):

    metrics_list = [[] for _ in range(len(metrics))]  # list of lists to store error metric results

    for j in range(len(metrics)):
        metrics_list[j].append(metrics[j](actual,predicted))

    df_metrics = pd.DataFrame({"metrics":metrics,"metrics_values":metrics_list})
    wandb.log({"table":df_metrics})
    return metrics_list


if __name__ == '__main__':
    with open("train/spectral_model_config_local.yaml", "r") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    wandb.login()
    wandb.init(project="syn_model_evaluation")
    config = wandb.config
    config.train_size = train_config["train_size"]
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

    model = models.spectralGP_model.SpectralMixtureGPModel(X_train, y_train, likelihood, train_config["mixtures"],config.num_dims)

    model.load_state_dict(torch.load( train_config["model_checkpoint_folder"] + "/spectral_model_training_size_" +
               str(config.train_size) + "_model_checkpoint.pt"))

    model.eval()

    #generrate predictions
    observed_pred = likelihood(model(torch.tensor(dfsubset.index, dtype=torch.float32)))
    print(observed_pred)

    #plot inference
    plot_inference(dfsubset, X_test, y_test, X_train, y_train)

    metrics = [me, rae, mape, rmse,mda] #list of metrics to compute see forecasting_metrics.p
    actual = y_test.numpy()
    predicted = observed_pred[len(X_train):].mean.detach().numpy()
    print(len(actual))
    print(len(predicted))

    result = compute_metrics(metrics,actual,predicted)
    print(result)

