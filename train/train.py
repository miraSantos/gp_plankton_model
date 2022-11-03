import math
import pandas as pd
import torch
import gpytorch
import os
import sys

sys.path.append(os.getcwd())

from tqdm import tqdm
import yaml
import models.spectralGP_model
from train.train_utils import *

import wandb  # library for tracking and visualization
wandb.login()



if __name__ == '__main__':
    with open("train/spectral_model_config.yaml", "r") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    wandb.login()

    slurm_id = sys.argv[1]
    num_mixtures = sys.argv[2]
    print("slurm_id"+slurm_id)
    print("num_mixtures" + num_mixtures)

    # wandb.init(project="syn_model_slurm_spectral_only")
    wandb.init(project="syn_model_slurm_spectral_only",mode = "disabled")

    config = wandb.config
    config.train_size = int(slurm_id) / 10  # passing thru slurm id to parallelize train size
    config.mixtures = int(num_mixtures)
    config.learning_rate = train_config["learning_rate"]
    config.predictor = train_config["predictor"]
    config.dependent = train_config["dependent"]
    config.better_lower_bound = train_config["better_lower_bound"]
    config.num_dims = train_config["num_dims"]

    df = pd.read_csv(train_config["data_path"])

    #loading in dataframe and assigning predictor and dependent variables
    df = pd.read_csv(train_config["data_path"])
    print(df.head())
    dfsubset = df.dropna(subset=config.dependent) #dropping na values #TODO: fix spectral model so that it can handle missing observations
    X = torch.tensor(dfsubset.index, dtype=torch.float32)
    y = torch.tensor(np.log(dfsubset[config.dependent].values), dtype=torch.float32)

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
    config.noise_value = train_config["noise_value"]
    config.noise_prior_loc = train_config["noise_prior_loc"]
    config.noise_prior_scale = train_config["noise_prior_scale"]
    config.res_path = train_config["res_path"]
    config.training_iter = train_config["training_iter"]

    plot_train_test_data(dfsubset, X_train, y_train, X_test, y_test, config)

    likelihood = gpytorch.likelihoods.GaussianLikelihoodWithMissingObs(
        noise_prior=gpytorch.priors.NormalPrior(config.noise_prior_loc, config.noise_prior_scale))
    model = models.spectralGP_model.SpectralMixtureGPModel(X_train, y_train, likelihood, config.mixtures,
                                                           config.num_dims)

    wandb.watch(model, log="all")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_model(likelihood, model, optimizer, config, X_train, y_train, learning_rate=config.learning_rate)

    # saving model checkpoint
    torch.save(model.state_dict(), train_config["model_checkpoint_folder"] + "/spectral_model_training_size_" +
               str(config.train_size) + "_model_checkpoint.pt")
