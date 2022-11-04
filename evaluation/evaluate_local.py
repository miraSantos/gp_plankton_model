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
from utils.eval import *

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

    df.loc[:,'date'] = pd.to_datetime(df.loc[:,'date'], format="%Y-%m-%d")
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
    ax[0].set_title("Evaluation " + "Training Size " + str(config["parameters"]["train_size"]*100) + "% of data")


    #plotting with DOY on the x-axis
    ax[1].scatter(df.doy_numeric[len(X_train):], y_test, label="observations",c = "green")
    ax[1].scatter(df.doy_numeric[len(X_train):], observed_pred.mean.detach().numpy()[len(X_train):],label = "prediction",c = "blue")
    ax[1].set_xlabel("Day of the Year")
    ax[1].set_ylabel("Syn Conc")
    ax[1].legend()
    eval_img = config["res_path"] +"/eval_train_size_" + str(config["parameters"]["train_size"]) + '.png'
    fig.savefig(eval_img)
    im = Image.open(eval_img)
    wandb.log({"Evaluation": wandb.Image(im)})




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="specify path to configuration file (yaml) ", type=str,
                        default="cfg/local_config.yaml")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    wandb.login()
    wandb.init(project="syn_model_evaluation", config=config, mode=config["wandb_mode"])



    df = pd.read_csv(config["data_path"], low_memory=False)
    df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'],
                                       format="%Y-%m-%d")  # required or else dates start at 1971! (WEIRD BUG HERE)
    dfsubset = df.dropna(subset=[config["dependent"], config["predictor"]])  # dropping na values #TODO: fix spectral model so that it can handle missing observations

    X = torch.load( config["split_folder"] + config["dependent"] + "X_dataset.pt")
    X_train = torch.load(config["split_folder"] + config["dependent"] + "train_size_" + str(config["parameters"]["train_size"]) + "_X_train.pt")
    y_train = torch.load(config["split_folder"] + config["dependent"] + "train_size_" + str(config["parameters"]["train_size"]) + "_y_train.pt")
    X_test = torch.load(config["split_folder"] + config["dependent"] + "train_size_" + str(config["parameters"]["train_size"]) + "_X_test.pt")
    y_test = torch.load(config["split_folder"] + config["dependent"] + "train_size_" + str(config["parameters"]["train_size"]) + "_y_test.pt")

    likelihood = gpytorch.likelihoods.GaussianLikelihoodWithMissingObs(noise_prior=gpytorch.priors.NormalPrior(config["parameters"]["noise_prior_loc"], config["parameters"]["noise_prior_scale"]))
    model = models.spectralGP_model.SpectralMixtureGPModel(X_train, y_train, likelihood, config["parameters"]["mixtures"], config["parameters"]['num_dims'])

    model_save_path = config["model_checkpoint_folder"] + "/spectral_model_training_size_" + str(config["parameters"]["train_size"]) + "_model_checkpoint.pt"

    model.load_state_dict(torch.load( model_save_path))

    model.eval()

    #generrate predictions

    observed_pred = likelihood(model(torch.tensor(X, dtype=torch.float32)))
    print(observed_pred)

    #plot inference
    plot_inference(dfsubset, X_test, y_test, X_train, y_train)

    metrics = [me, rae, mape, rmse,mda] #list of metrics to compute see forecasting_metrics.p
    actual = y_test.numpy()
    predicted = observed_pred[len(X_train):].mean.detach().numpy()
    print(len(actual))
    print(len(predicted))
    result = compute_metrics(metrics,actual,predicted)
    wandb.log({"result":result})

