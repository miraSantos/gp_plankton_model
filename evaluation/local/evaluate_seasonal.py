import os,sys, gpytorch, torch, yaml,argparse
sys.path.append(os.getcwd())

from evaluation.forecasting_metrics import *
from utils.train_utils import *
from utils.eval import *

import models.seasonalGP_model
import models.exactGP_model
import seaborn as sns
from tqdm import tqdm

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

    print("date", dfsubset.date[0:2005].shape)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
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

    observed_pred = likelihood(model(torch.tensor(X, dtype=torch.float32)))
    dfsubset["predictions"] = observed_pred.mean.detach().numpy()
    dfsubset["month"] = dfsubset.date.dt.month

    f_pred = model(torch.tensor(X,dtype=torch.float32))
    plt.plot(f_pred.mean.detach().numpy())
    plt.show()
    plt.savefig("results/seasonal_prediction.png")
    print("f_pred")
    print("f_pred mean",f_pred.mean)
    print(f_pred.covariance_matrix)
    print(f_pred)

    actual = y_test.numpy()
    predicted = observed_pred[len(X_train):].mean.detach().numpy()

    plot_evaluation(dfsubset,config,actual,predicted,observed_pred,y_train,X,y_test,X_train)

    metrics = [me, rae, mape, rmse,mda] #list of metrics to compute see forecasting_metrics.p
    result = compute_metrics(metrics,actual,predicted)
    wandb.log({"result":result})


    print("mean error: "+ str(me(actual , predicted)))
    print("mean average percentage error: ", str(mape(actual,predicted)))
    print("relative absolute error: ", str(rae(actual,predicted)))
    print("mean directional accuracy: ", str(mda(actual,predicted)))
    # print("mean average scaled error" , str(mase(actual,predicted,seasonality=360)))

    plt.close()
    plt.close()


