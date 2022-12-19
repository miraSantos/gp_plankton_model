import os, sys, argparse
sys.path.append(os.getcwd())

import yaml
from utils.train_utils import *
import wandb  # library for tracking and visualization
import models.spectralGP_model


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
    X = torch.linspace(0,1,len(dfsubset.index))
    if config["take_log"]==True:
        dependent = np.log(dfsubset[config["dependent"]].values)
    else:
        dependent = dfsubset[config["dependent"]].values
    y = torch.tensor(dependent, dtype=torch.float32)

    # #defining training data based on testing split
    X_train, y_train, X_test, y_test = define_training_data(X, y, train_size=config["train_size"], normalize=wandb.config.normalize)

    print(X_train)
    print(y_train)
    plot_train_test_data(dfsubset, X_train, y_train, X_test, y_test, config)

    # likelihood = gpytorch.likelihoods.GaussianLikelihoodWithMissingObs(noise_prior=gpytorch.priors.NormalPrior(config["noise_prior_loc"], config["noise_prior_scale"]))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = models.spectralGP_model.SpectralMixtureGPModel(X_train, y_train, likelihood,
                                                           config["mixtures"],
                                                           config['num_dims']
                                                           )

    wandb.watch(model, log="all")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    train_model(likelihood, model, optimizer, X_train, y_train)

    # # saving model checkpoint
    model_save_path = config["model_save_path"]
    torch.save(model.state_dict(), model_save_path)
    wandb.save(model_save_path)