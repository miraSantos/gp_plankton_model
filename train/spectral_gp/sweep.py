import os, sys, argparse

sys.path.append(os.getcwd())

from utils.train_utils import *
from utils.eval import *
from evaluation.forecasting_metrics import *
import torch
import gpytorch
import models.spectralGP_model


def main_sweep():
    likelihood = gpytorch.likelihoods.GaussianLikelihoodWithMissingObs(
        noise_prior=gpytorch.priors.NormalPrior(wandb.config.noise_prior_loc,
                                                wandb.config.noise_prior_scale))
    model = models.spectralGP_model.SpectralMixtureGPModel(X_train, y_train,
                                                           likelihood,
                                                           wandb.config.mixtures,
                                                           wandb.config.num_dims,
                                                           config["mixture_scales_prior"],
                                                           config["mixture_means_prior"],
                                                           config["mixture_weights_prior"]
                                                           )

    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)
    wandb.watch(model, log="all")

    print("training started")
    train_model(likelihood, model, optimizer, X_train, y_train)

    print("training finished")

    model_save_path = config["model_checkpoint_folder"] + "/spectral_model_training_size_" + str(
        wandb.config.train_size) + "_model_checkpoint.pt"
    torch.save(model.state_dict(), model_save_path)
    wandb.save(model_save_path)

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X))
    actual = y_test.numpy()
    predicted = observed_pred[len(X_train):].mean.detach().numpy()

    # print(actual)
    # print(predicted)
    plot_inference(dfsubset, y_test, X_train, y_train, observed_pred)

    metrics = [me, rae, mape, rmse, mda]  # list of metrics to compute see forecasting_metrics.py
    result = compute_metrics(metrics, actual, predicted)
    wandb.log({"result": result})

    wandb.log({
        'mean_error': me(actual, predicted),
        'rel_abs_error': rae(actual, predicted),
        'mean_avg_per_error': mape(actual, predicted),
        'rt_sq_mean_error': rmse(actual, predicted),
        'mean_dir_acc': mda(actual, predicted),
        'mean_absolute_scaled_error': mase(actual, predicted, wandb.config.seasonality)
    })


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="specify path to configuration file (yaml) ", type=str,
                        default="cfg/local_config.yaml")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    wandb.login()

    run = wandb.init(mode=config["wandb_mode"])

    # loading data
    dfsubset, X, X_train, y_train, X_test, y_test = load_test_train(config)
    plot_train_test_data(dfsubset, X_train, y_train, X_test, y_test, config)
    # # saving model checkpoint
    main_sweep()
