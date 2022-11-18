from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel
from sklearn import preprocessing

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from calendar import month_name as mn
import datetime
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.dates as mdates #for working with dates in plots
from tqdm import tqdm



############################
#Importing Data
#############################

PATH = "../datasets/df_merged_daily_phyto_2022-10-21.csv"
df = pd.read_csv(PATH)#.drop_duplicates(subset = "date") #remove duplicate days


months = mn[1:] #importing month names from calendar

#reformat dates to datetime format
df.date = pd.to_datetime(df.date, format = "%Y-%m-%d")
df.year = df.date.dt.year
df.month = df.date.dt.month
df.doy_numeric = df.date.dt.dayofyear
df.doy = pd.to_datetime(df.date.dt.dayofyear, format = "%j")
df.monthname = pd.Categorical(df.date.dt.month_name(), categories=months, ordered = True)
df.head()

df.describe().transpose()

#Preparing data for model
dfgp = df[["year","doy_numeric","synconc"]].dropna()
X = (dfgp.year + dfgp.doy_numeric/365).to_numpy().reshape(-1,1)
y = np.log(dfgp.synconc).to_numpy()
y_mean = y.mean()
# y = preprocessing.normalize([synconc])

############################
#Kernel Design
#############################

# Long Term Trend
long_term_trend_kernel = 2.0 ** 2 * RBF(length_scale=1.0)

#Seasonal Trend
seasonal_kernel = (
    2.0 ** 2
    * RBF(length_scale=100.0)
    * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds=[0,1])
)

#Smaller Irregularities
irregularities_kernel = 0.5 ** 2 * RationalQuadratic(length_scale=1.0, alpha=1.0)

#White Noise
noise_kernel = 0.1 ** 2 * RBF(length_scale=0.1) + WhiteKernel(
    noise_level=0.1 ** 2, noise_level_bounds=(1e-5, 1e5)
)
#Final Kernel
final_kernel = (
    long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel
)
final_kernel

#################################
#Gaussian Process Fitting
##################################

gaussian_process = GaussianProcessRegressor(kernel = final_kernel, normalize_y = False)

end = 1000
for ee in tqdm(range(3,end)):
    gaussian_process.fit(X[1:ee],y[1:ee]-y_mean)


    today = datetime.datetime.now()
    current_day = today.year + today.day / 365
    X_test = np.linspace(start=2003, stop = 2018, num=1_000).reshape(-1, 1)
    mean_y_pred, std_y_pred = gaussian_process.predict(X_test, return_std=True)
    mean_y_pred += y_mean

    # mean_y_pred = np.exp(mean_y_pred)
    # std_y_pred = np.exp(std_y_pred)

    plt.figure(figsize=(15, 8))
    plt.scatter(X[1:ee], y[1:ee],color="black",marker = "x", label="Measurements")
    plt.plot(X_test, mean_y_pred, color="tab:blue", alpha=0.4, label="Gaussian process")
    plt.fill_between(
        X_test.ravel(),
        mean_y_pred - std_y_pred,
        mean_y_pred + std_y_pred,
        color="tab:blue",
        alpha=0.2,
    )
    plt.ylim([0,13])
    plt.legend()
    plt.xlabel("Year")
    plt.ylabel("Log(Daily Average of Syn Concentration)")
    _ = plt.title(
        "Daily Average of Syn Concentration"
    )
    
    plt.savefig("../results/sci_kit_learn_results/" + str(ee).zfill(3) + "gpsyn.jpg")
    plt.close()

    plt.figure(figsize=(15, 8))
    plt.scatter(df.date[1:ee], y[1:ee], color="black", marker="x", label="Measurements")
    plt.plot(df.date[-len(X_test):],mean_y_pred, color="tab:blue", alpha=0.4, label="Gaussian process")
    plt.fill_between(
        X_test.ravel(),
        mean_y_pred - std_y_pred,
        mean_y_pred + std_y_pred,
        color="tab:blue",
        alpha=0.2,
    )
    plt.ylim([0, 13])
    plt.legend()
    plt.xlabel("Day of Year")
    plt.ylabel("Log(Daily Average of Syn Concentration)")
    _ = plt.title(
        "Daily Average of Syn Concentration"
    )

    plt.savefig("../results/sci_kit_learn_results/" + str(ee).zfill(3) + "gpsyn_doy.jpg")
    plt.close()


