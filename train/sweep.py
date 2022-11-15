import os, sys, argparse
sys.path.append(os.getcwd())

from utils.train_utils import *
import wandb  # library for tracking and visualization
from utils.eval import *
from evaluation.forecasting_metrics import *


def plot_inference(df,X_test, y_test, X_train, y_train,observed_pred):
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
    ax[0].set_title("Evaluation " + "Training Size " + str(wandb.config.train_size*100) + "% of data")


    #plotting with DOY on the x-axis
    ax[1].scatter(df.doy_numeric[len(X_train):], y_test, label="observations",c = "green")
    ax[1].scatter(df.doy_numeric[len(X_train):], observed_pred.mean.detach().numpy()[len(X_train):],label = "prediction",c = "blue")
    ax[1].set_xlabel("Day of the Year")
    ax[1].set_ylabel("Syn Conc")
    ax[1].legend()
    eval_img = config["res_path"] + "/" + config["dependent"]+"/eval_train_size_" + str(wandb.config.train_size) + '.png'
    fig.savefig(eval_img)
    wandb.save(eval_img)
    im = Image.open(eval_img)

    wandb.log({"Evaluation": wandb.Image(im)})



def load_test_train():
    df = pd.read_csv(config["data_path"], low_memory=False)
    df.loc[:,'date'] = pd.to_datetime(df.loc[:,'date'], format="%Y-%m-%d") #required or else dates start at 1971! (WEIRD BUG HERE)
    dfsubset = df.dropna(subset=config["dependent"]) #dropping na values #TODO: fix spectral model so that it can handle missing observations

    if wandb.config.num_dims > 1:
        dfsubset = df.dropna(subset=[config["dependent"], config["predictor"]])  # dropping na values #TODO: fix spectral model so that it can handle missing observations
        X = torch.tensor(dfsubset.loc[:, config["predictor"]].reset_index().to_numpy(),
                         dtype=torch.float32)  # 2D tensor
    elif wandb.config.num_dims == 1:
        X = torch.tensor(dfsubset.index, dtype=torch.float32)
    else:
        assert value > 0 , f"number greater than 0 expected, got: {wandb.config.num_dims}"

    if config["take_log"]:
        dependent = np.log(dfsubset[config["dependent"]].values)
    else:
        dependent = dfsubset[config["dependent"]].values
    y = torch.tensor(dependent, dtype=torch.float32)

    # #defining training data based on testing split
    X_train, y_train, X_test, y_test = define_training_data(X, y, train_size=wandb.config.train_size, normalize=True)

    torch.save(X, config["split_folder"] + config["dependent"] + "X_dataset.pt")
    torch.save(X_train, config["split_folder"] + config["dependent"] + "train_size_" + str(wandb.config.train_size) + "_X_train.pt")
    torch.save(y_train, config["split_folder"] + config["dependent"] + "train_size_" + str(wandb.config.train_size) + "_y_train.pt")
    torch.save(X_test, config["split_folder"] + config["dependent"] + "train_size_" + str(wandb.config.train_size) + "_X_test.pt")
    torch.save(y_test, config["split_folder"] + config["dependent"] + "train_size_" + str(wandb.config.train_size) + "_y_test.pt")

    wandb.config.X_train_shape = X_train.shape
    wandb.config.y_train_shape = y_train.shape
    wandb.config.X_test_shape = X_test.shape
    wandb.config.y_test_shape = y_test.shape

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)

    return dfsubset, X, X_train, y_train, X_test, y_test

def main_sweep():

    likelihood = gpytorch.likelihoods.GaussianLikelihoodWithMissingObs(
        noise_prior=gpytorch.priors.NormalPrior(wandb.config.lr,
                                                wandb.config.noise_prior_scale))
    model = models.spectralGP_model.SpectralMixtureGPModel(X_train, y_train, likelihood,
                                                           wandb.config.mixtures,
                                                           wandb.config.num_dims)

    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)
    wandb.watch(model, log="all")
    train_model(likelihood, model, optimizer, X_train, y_train)

    print("training finished")

    model_save_path = config["model_checkpoint_folder"] + "/spectral_model_training_size_" + str(wandb.config.train_size) + "_model_checkpoint.pt"
    torch.save(model.state_dict(), model_save_path)
    wandb.save(model_save_path)

    model.eval()

    observed_pred = likelihood(model(torch.tensor(X, dtype=torch.float32)))
    actual = y_test.numpy()
    predicted = observed_pred[len(X_train):].mean.detach().numpy()

    print(actual)
    print(predicted)
    plot_inference(dfsubset, X_test, y_test, X_train, y_train,observed_pred)

    metrics = [me, rae, mape, rmse,mda] #list of metrics to compute see forecasting_metrics.p
    result = compute_metrics(metrics,actual,predicted)
    wandb.log({"result" : result})

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

    #loading data
    dfsubset, X, X_train, y_train, X_test, y_test = load_test_train()
    plot_train_test_data(dfsubset, X_train, y_train, X_test, y_test, config)
    # # saving model checkpoint
    main_sweep()
