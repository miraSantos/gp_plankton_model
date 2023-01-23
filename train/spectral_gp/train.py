import os, sys, argparse, math, torch
sys.path.append(os.getcwd())

from utils.train_utils import *
import wandb  # library for tracking and visualization

from matplotlib import pyplot as plt

from models.sebastian_spectral import SMKernelGP
# from smk.train import train
# from smk.plots import plot_density, plot_kernel, save_plot



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="specify path to configuration file (yaml) ", type=str,
                        default="cfg/local_config.yaml")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    wandb.login()
    wandb.init(project="testing",config=config,mode=config["wandb_mode"])

    df = pd.read_csv(config["data_path"], low_memory=False)
    df.loc[:,'date'] = pd.to_datetime(df.loc[:,'date'], format="%Y-%m-%d") #required or else dates start at 1971! (WEIRD BUG HERE)
    dfsubset = df.dropna(subset=config["dependent"]) #dropping na values #TODO: fix spectral model so that it can handle missing observations
    X = torch.tensor(dfsubset.index, dtype=torch.float32)
    if config["take_log"]==True:
        dependent = np.log(dfsubset[config["dependent"]].values)
    else:
        dependent = dfsubset[config["dependent"]].values
    y = torch.tensor(dependent, dtype=torch.float32)

    # #defining training data based on testing split
    X_train, y_train, X_test, y_test = define_training_data(X, y, train_size=config["parameters"]["train_size"], normalize=True)


    plot_train_test_data(dfsubset, X_train, y_train, X_test, y_test, config)

    likelihood = gpytorch.likelihoods.GaussianLikelihoodWithMissingObs(noise_prior=gpytorch.priors.NormalPrior(config["parameters"]["noise_prior_loc"], config["parameters"]["noise_prior_scale"]))
    model = models.spectralGP_model.SpectralMixtureGPModel(X_train, y_train, likelihood, config["parameters"]["mixtures"], config["parameters"]['num_dims'])

    wandb.watch(model, log="all")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["parameters"]["lr"])
    train_model(likelihood, model, optimizer, config, X_train, y_train, learning_rate=config["parameters"]["lr"])

    # # saving model checkpoint
    model_save_path = config["model_checkpoint_folder"] + "/spectral_model_training_size_" + str(config["train_size"]) + "_model_checkpoint.pt"
    torch.save(model.state_dict(), model_save_path)
    wandb.save(model_save_path)