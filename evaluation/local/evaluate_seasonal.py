import os
import sys

sys.path.append(os.getcwd())

import gpytorch
import torch
import yaml
from evaluation.forecasting_metrics import *
from utils.eval import *
from utils.train_utils import *

import models.seasonalGP_model
import models.exactGP_model
import argparse
import seaborn as sns

def plot_evaluation(dfsubset,config,actual,predicted):
    # FULL TIMESERIES and DOY Plot
    width = 20
    height = 8
    fig, ax = plt.subplots(1, 2, figsize=(width, height))
    # Training Data
    ax[0].scatter(dfsubset.date[0:len(y_train)], y_train.numpy(),
                  label="Observations",
                  c="blue",
                  marker="o",
                  facecolors='none')
    # Testing Data
    ax[0].scatter(dfsubset.date[len(y_train):], y_test.numpy(),
                  c="mediumseagreen",
                  marker="+",
                  label="Testing data")
    # Prediction
    ax[0].plot(dfsubset.date[0:len(X)],
               observed_pred.mean.detach().numpy(),
               label="prediction",
               c="red")
    ax[0].set_title("Seasonal Kernel forecast: " + str(config["dependent"]))
    ax[0].set_ylabel(config["dependent"])
    ax[0].set_xlabel("Year")
    ax[0].legend()
    ax[0].grid()
    ax[1].scatter(dfsubset.doy_numeric[len(y_train):], actual, c="mediumseagreen", label="Observations")
    ax[1].scatter(dfsubset.doy_numeric[len(y_train):], predicted, c="red", label="Predictions")
    ax[1].set_ylabel(config["dependent"])
    ax[1].set_xlabel("Day of Year")
    ax[1].set_title("Prophet forecast by day of year: " + str(config["dependent"]))
    ax[1].legend()
    ax[1].grid()
    plt.show()

    eval_img = config["res_path"] + "/" + config["dependent"] + "/full_timeseries_train_size_" + str(
        config["train_size"]) + '.png'
    fig.savefig(eval_img)
    wandb.save(eval_img)
    im = Image.open(eval_img)
    wandb.log({"Full Timeseries Evaluation": wandb.Image(im)})

    #
    # Actual vs. Predicted and Violin Plot
    new_long = pd.melt(dfsubset[["date", "month", config["dependent"], "predictions"]].loc[len(X_train):],
                       id_vars=["month", "date"], value_vars=[config["dependent"], "predictions"], value_name="conc")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].scatter(dfsubset.date[len(y_train):], actual, c="mediumseagreen", marker="x", label="observations")
    ax[0].plot(dfsubset.date[len(y_train):], predicted, c="red", label="predictions")
    ax[0].legend()
    ax[0].set_ylabel(config["dependent"])
    ax[0].set_xlabel("Year")
    ax[0].set_title("Observations vs. Predictions for Testing")
    pal = {config["dependent"]: "mediumseagreen", "predictions": "r"}
    sns.violinplot(axes=ax[1], data=new_long, x="month", y="conc", hue="variable", split=True, palette=pal)
    # ax[1].legend(handles=ax.legend_.legendHandles, labels=["Observations", "Predictions"])
    ax[1].set_title("Violin Plot of Predictive Check")
    # ax.set_ylim(0,14)
    ax[1].grid()
    plt.show()

    eval_img = config["res_path"] + "/" + config["dependent"] + "/violin_train_size_" + str(
        config["train_size"]) + '.png'
    fig.savefig(eval_img)
    wandb.save(eval_img)
    im = Image.open(eval_img)
    wandb.log({"Violin Plot": wandb.Image(im)})

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

    print("date", dfsubset.date[0:2005].shape)

    likelihood = gpytorch.likelihoods.GaussianLikelihoodWithMissingObs(noise_prior=gpytorch.priors.NormalPrior(config["noise_prior_loc"], config["noise_prior_scale"]))
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

    actual = y_test.numpy()
    predicted = observed_pred[len(X_train):].mean.detach().numpy()

    plot_evaluation(dfsubset,config,actual,predicted)

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


