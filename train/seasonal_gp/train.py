import os, sys, argparse, yaml, wandb, gpytorch
sys.path.append(os.getcwd())
import models.seasonalGP_model
import models.exactGP_model
from utils.train_utils import *
from utils.eval import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="specify path to configuration file (yaml) ", type=str,
                        default="cfg/local_config.yaml")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    wandb.login()
    wandb.init(project=config["project"], config=config, mode=config["wandb_mode"])

    df = pd.read_csv(config["data_path"], low_memory=False)
    df.loc[:,'date'] = pd.to_datetime(df.loc[:,'date'], format="%Y-%m-%d") #required or else dates start at 1971! (WEIRD BUG HERE)
    df.loc[:,"doy_numeric"] = df.date.dt.dayofyear
    dfsubset = df.dropna(subset=(config["dependent"])) #dropping na values #TODO: fix spectral model so that it can handle missing observations
    X = torch.linspace(0,1,len(dfsubset.index))
    if config["take_log"]==True:
        dfsubset[config["dependent"]] = np.log(dfsubset[config["dependent"]])
    elif config["take_cube_root"]==True:
        dfsubset[config["dependent"]] = np.power(dfsubset[config["dependent"]],1/3)

    dependent = dfsubset[config["dependent"]].values
    y = torch.tensor(dependent, dtype=torch.float32)

    # #defining training data based on testing split
    X_train, y_train, X_test, y_test = define_training_data(X, y, train_size=config["train_size"], normalize=wandb.config.normalize)

    plot_train_test_data(dfsubset, X_train, y_train, X_test, y_test, config)

    print(X_train)
    print(y_train)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # likelihood = gpytorch.likelihoods.GaussianLikelihoodWithMissingObs(noise_prior=gpytorch.priors.NormalPrior(config["noise_prior_loc"],
    #                                                                   config["noise_prior_scale"]))
    # model = models.seasonalGP_model.seasonalGPModel(X_train, y_train,
    #                                                 likelihood,
    #                                                 wandb.config.num_dims,
    #                                                 exec(wandb.config.lt_t_prior),
    #                                                 exec(wandb.config.lt_l_constraint),
    #                                                 wandb.config.lt_eps,
    #                                                 exec(wandb.config.s_rbf_l_prior),
    #                                                 exec(wandb.config.s_rbf_l_constraint),
    #                                                 wandb.config.s_rbf_eps,
    #                                                 exec(wandb.config.s_pl_prior),
    #                                                 exec(wandb.config.s_pl_constraint),
    #                                                 exec(wandb.config.s_pk_l_prior),
    #                                                 exec(wandb.config.s_pk_l_constraint),
    #                                                 wandb.config.s_pk_eps,
    #                                                 exec(wandb.config.wn_l_prior),
    #                                                 exec(wandb.config.wn_l_constraint),
    #                                                 exec(wandb.config.wn_a_constraint),
    #                                                 wandb.config.wn_eps
    #                                                 )
    model = models.exactGP_model.exactGP_model(X_train,y_train,likelihood)

    wandb.watch(model, log="all")
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)
    train_model(likelihood, model, optimizer, X_train, y_train)

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 2, len(dfsubset.date))
        observed_pred = likelihood(model(test_x))
    plt.plot(dfsubset.date,observed_pred.mean.detach().numpy())
    plt.show()

    # # saving model checkpoint
    model_save_path = config["model_save_path"]
    torch.save(model.state_dict(), model_save_path)
    wandb.save(model_save_path)