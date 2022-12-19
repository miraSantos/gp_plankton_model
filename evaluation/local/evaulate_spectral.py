import os,sys, gpytorch, torch, yaml, argparse
sys.path.append(os.getcwd())

from tqdm import tqdm
from evaluation.forecasting_metrics import *
from utils.eval import *
from utils.train_utils import *

import models.seasonalGP_model
import models.exactGP_model
import models.spectralGP_model
import seaborn as sns


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
    df.loc[:,'date'] = pd.to_datetime(df.loc[:,'date'], format="%Y-%m-%d") #required or else dates start at 1971! (WEIRD BUG HERE)
    df.loc[:,"doy_numeric"] = df.date.dt.dayofyear
    dfsubset = df.dropna(subset=config["dependent"]) #dropping na values #TODO: fix spectral model so that it can handle missing observations
    print(dfsubset.head())
    X = torch.linspace(0,1,len(dfsubset.date))
    if config["take_log"]==True:
        dependent = np.log(dfsubset[config["dependent"]].values)
    else:
        dependent = dfsubset[config["dependent"]].values
    y = torch.tensor(dependent, dtype=torch.float32)

    print("dfsubset shape",dfsubset.shape)

    # #defining training data based on testing split
    X_train, y_train, X_test, y_test = define_training_data(X, y, train_size=config["train_size"], normalize=wandb.config.normalize)

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = models.spectralGP_model.SpectralMixtureGPModel(X_train, y_train, likelihood,
                                                           config["mixtures"], config['num_dims'])


    model_save_path = config["model_save_path"]

    model.load_state_dict(torch.load(model_save_path))

    model.eval()
    likelihood.eval()

    # f_preds = model(X_test)
    #
    # f_mean = f_preds.mean
    # f_var = f_preds.variance
    # f_covar = f_preds.covariance_matrix
    #
    # print(f_covar)

    #generrate predictions

    observed_pred = tqdm(likelihood(model(X)))
    # dfsubset["predictions"] = observed_pred.mean.detach().numpy()
    # dfsubset["month"] = dfsubset.date.dt.month
    #
    # actual = y_test.numpy()
    # predicted = observed_pred[len(X_train):].mean.detach().numpy()
    #
    # plot_evaluation(dfsubset,config,actual,predicted)
    #
    # metrics = [me, rae, mape, rmse,mda] #list of metrics to compute see forecasting_metrics.p
    # result = compute_metrics(metrics,actual,predicted)
    # wandb.log({"result":result})
    #
    #
    # print("mean error: "+ str(me(actual , predicted)))
    # print("mean average percentage error: ", str(mape(actual,predicted)))
    # print("relative absolute error: ", str(rae(actual,predicted)))
    # print("mean directional accuracy: ", str(mda(actual,predicted)))
    # # print("mean average scaled error" , str(mase(actual,predicted,seasonality=360)))
    #
    # plt.close()
    # plt.close()


