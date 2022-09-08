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

import models.spectralGP_model
import argparse

def plot_inference(X_test, y_test, X_train, y_train):
    """
    :param x_test:
    :param likelihood:
    :param model:
    :return:
    """
    # Initialize plot

    df = pd.read_csv(train_config["data_path"])
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    width = 20
    height = 5
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training preprocess as black stars
    ax.plot(df.date[:len(X_train)], y_train, 'k*', label="training data")
    # Plot predictive means as blue line
    ax.plot(df.date[:len(X_test)], observed_pred.mean.detach().numpy(), 'b', label="prediction")
    ax.plot(df.date[len(X_train):], y_test, 'g', label="testing data")
    # Shade between the lower and upper confidence bounds
    ax.fill_between(df.date[:len(X_test)], lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Year")
    ax.set_ylabel("[log(Syn)] (Normalized")
    ax.legend()
    ax.grid()
    eval_img = train_config["res_path"] +"/eval_train_size_" + str(args.train_size) + '.png'
    ax.set_title("Evaluation " + "Training Size " + str(args.train_size*100) + "% of data")
    fig.savefig(eval_img)
    im = Image.open(eval_img)
    wandb.log({"Evaluation": wandb.Image(im)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training size expressed as a fraction')
    parser.add_argument('--train_size', type=float
                        )
    args = parser.parse_args()

    with open("train/train_config.yaml", "r") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    wandb.login()

    wandb.init(project="syn_model")
    config = wandb.config
    config.train_size = args.train_size
    config.num_mixtures = train_config["num_mixtures"]
    config.learning_rate = train_config["learning_rate"]
    config.predictor = 'daily_index'
    config.dependent = train_config["dependent"]

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    X_train = torch.load(train_config["split_folder"] + "train_size_" + str(args.train_size) + "_X_train.pt")
    y_train = torch.load(train_config["split_folder"] + "train_size_" + str(args.train_size) + "_y_train.pt")
    X_test = torch.load(train_config["split_folder"] + "train_size_" + str(args.train_size) + "_X_test.pt")
    y_test = torch.load(train_config["split_folder"] + "train_size_" + str(args.train_size) + "_y_test.pt")

    model = models.spectralGP_model.SpectralMixtureGPModel(X_train, y_train, likelihood, train_config["num_mixtures"])
    model.load_state_dict(torch.load(train_config["model_checkpoint_folder"] + "/training_size_" +
                                     str(args.train_size) + "_model_checkpoint.pt"))
    model.eval()

    observed_pred = likelihood(model(X_test))
    print(observed_pred)
    plot_inference(X_test, y_test, X_train, y_train)

