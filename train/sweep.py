import os, sys, argparse
sys.path.append(os.getcwd())

from utils.train_utils import *
import wandb  # library for tracking and visualization
from utils.eval import *

def load_test_train():
    df = pd.read_csv(wandb.config.data_path, low_memory=False)
    df.loc[:,'date'] = pd.to_datetime(df.loc[:,'date'], format="%Y-%m-%d") #required or else dates start at 1971! (WEIRD BUG HERE)
    dfsubset = df.dropna(subset=wandb.config.dependent) #dropping na values #TODO: fix spectral model so that it can handle missing observations

    if wandb.config.num_dims_predictor >= 1:
        X = torch.tensor(dfsubset.loc[:, wandb.config.predictor].reset_index().to_numpy(),
                         dtype=torch.float32)  # 2D tensor
    else:
        X = torch.tensor(dfsubset.index, dtype=torch.float32)

    if wandb.config.take_log==True:
        dependent = np.log(dfsubset[wandb.config.dependent].values)
    else:
        dependent = dfsubset[wandb.config.dependent].values
    y = torch.tensor(dependent, dtype=torch.float32)

    # #defining training data based on testing split
    X_train, y_train, X_test, y_test = define_training_data(X, y, train_size=wandb.config.train_size, normalize=True)

    torch.save(X, wandb.config.split_folder + wandb.config.dependent + "X_dataset.pt")
    torch.save(X_train, wandb.config.split_folder + wandb.config.dependent + "train_size_" + str(wandb.config.train_size) + "_X_train.pt")
    torch.save(y_train, wandb.config.split_folder + wandb.config.dependent + "train_size_" + str(wandb.config.train_size) + "_y_train.pt")
    torch.save(X_test, wandb.config.split_folder + wandb.config.dependent + "train_size_" + str(wandb.config.train_size) + "_X_test.pt")
    torch.save(y_test, wandb.config.split_folder + wandb.config.dependent + "train_size_" + str(wandb.config.train_size) + "_y_test.pt")

    wandb.config.X_train_shape = X_train.shape
    wandb.config.y_train_shape = y_train.shape
    wandb.config.X_test_shape = X_test.shape
    wandb.config.y_test_shape = y_test.shape

    return dfsubset, X_train, y_train, X_test, y_test

def main_sweep():
    run = wandb.init(project="testing", mode="disabled")

    likelihood = gpytorch.likelihoods.GaussianLikelihoodWithMissingObs(
        noise_prior=gpytorch.priors.NormalPrior(wandb.config.lr,
                                                wandb.config.noise_prior_scale))
    model = models.spectralGP_model.SpectralMixtureGPModel(X_train, y_train, likelihood,
                                                           wandb.config.mixtures,
                                                           wandb.config.num_dims)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    wandb.watch(model, log="all")
    train_model(likelihood, model, optimizer, config, X_train, y_train,
                learning_rate=config.lr)

    model_save_path = wandb.config.model_checkpoint_folder + "/spectral_model_training_size_" + str(wandb.config.train_size) + "_model_checkpoint.pt"
    torch.save(model.state_dict(), model_save_path)
    wandb.save(model.state.dict())
    wandb.save(model_save_path)
    observed_pred = likelihood(model(torch.tensor(dfsubset.index, dtype=torch.float32)))
    actual = y_test.numpy()
    predicted = observed_pred[len(X_train):].mean.detach().numpy()

    plot_inference(dfsubset, X_test, y_test, X_train, y_train)

    metrics = [me, rae, mape, rmse,mda] #list of metrics to compute see forecasting_metrics.p
    result = compute_metrics(metrics,actual,predicted)
    wandb.log({"result":result})

    wandb.log({
        'mean_error': me(actual, predicted),
        'rel_abs_error': rae(actual, predicted),
        'mean_avg_per_error': mape(actual, predicted),
        'rt_sq_mean_error': rmse(actual, predicted),
        'mean_dir_acc': mda(actual, predicted)
    })

if __name__ == '__main__':

    #logging into wandb
    wandb.login()

    #loading data
    dfsubset, X_train, y_train, X_test, y_test = load_test_train()
    plot_train_test_data(dfsubset, X_train, y_train, X_test, y_test)
    # # saving model checkpoint
    main_sweep()
