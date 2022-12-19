
import pandas as pd
import os
import sys
import matplotlib.dates as mdates  # v 3.3.2

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

from PIL import Image
import wandb  # library for tracking and visualization
from evaluation.forecasting_metrics import *



def plot_evaluation(dfsubset,config,actual,predicted):
    # FULL TIMESERIES and DOY Plot
    lower, upper = observed_pred.confidence_region()
    print(type(observed_pred))
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
    ax[0].fill_between(dfsubset.date[0:len(X)], lower.detach().numpy(), upper.detach().numpy(), alpha=0.2,color = "red")
    ax[0].set_title("Seasonal Kernel forecast: " + str(config["dependent"]))
    ax[0].set_ylabel(config["dependent"])
    ax[0].set_xlabel("Year")
    ax[0].legend()
    ax[0].grid()
    ax[1].scatter(dfsubset.doy_numeric[len(y_train):], actual, c="mediumseagreen", label="Observations")
    ax[1].scatter(dfsubset.doy_numeric[len(y_train):], predicted, c="red", label="Predictions")
    ax[1].fill_between(dfsubset.doy_numeric[0:len(X)], lower.detach().numpy(), upper.detach().numpy(), alpha=0.2,color = "red")
    ax[1].set_ylabel(config["dependent"])
    ax[1].set_xlabel("Day of Year")
    ax[1].set_title("Forecast by day of year: " + str(config["dependent"]))
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
    ax[1].legend(handles=ax.legend_.legendHandles, labels=["Observations", "Predictions"])
    ax[1].set_title("Violin Plot of Predictive Check")
    ax[1].grid()
    plt.show()

    eval_img = config["res_path"] + "/" + config["dependent"] + "/violin_train_size_" + str(
        config["train_size"]) + '.png'
    fig.savefig(eval_img)
    wandb.save(eval_img)
    im = Image.open(eval_img)
    wandb.log({"Violin Plot": wandb.Image(im)})


def compute_metrics(metrics, actual, predicted):

    metrics_list = [[] for _ in range(len(metrics))]  # list of lists to store error metric results

    for j in range(len(metrics)):
        metrics_list[j].append(metrics[j](actual,predicted))

    df_metrics = pd.DataFrame({"metrics":metrics,"metrics_values":metrics_list})
    wandb.log({"table":df_metrics})
    return metrics_list