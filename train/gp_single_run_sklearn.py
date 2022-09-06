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



############################
#Importing Data
#############################

PATH = "../data/dfmerged_dailysynenv.csv"
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

dfgp = df[["year","doy_numeric","synconc"]].dropna() #isolating year, day of year and syn concentration
X = (dfgp.year + dfgp.doy_numeric/365).to_numpy().reshape(-1,1) #need to convert date into a number so calculating year + day of year and fitting to numpy array
y = np.log(dfgp.synconc).to_numpy()
y_mean = y.mean()
# y = preprocessing.normalize([synconc])

############################
#Kernel Design
#############################

long_term_trend_kernel = 2.0 ** 2 * RBF(length_scale=365.0)

seasonal_kernel = (
    2.0 ** 2
    * RBF(length_scale=100.0)
    * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds=[0,1])
)

irregularities_kernel = 0.5 ** 2 * RationalQuadratic(length_scale=1.0, alpha=1.0)

noise_kernel = 0.1 ** 2 * RBF(length_scale=0.1) + WhiteKernel(
    noise_level=0.1 ** 2, noise_level_bounds=(1e-5, 1e5)
)

final_kernel = (
    long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel
)
final_kernel

############################
#Gaussian Process Fitting
#############################

#building the Gaussian process regressor
gaussian_process = GaussianProcessRegressor(kernel = final_kernel , alpha = 0.5 ** 2)

#number of observations to include in the model
ee = len(y)

#
gaussian_process.fit(X[1:ee],y[1:ee]-y_mean)

#
today = datetime.datetime.now()
current_day = today.year + today.day / 365

X_test = np.linspace(start=2003, stop = current_day , num=1_000).reshape(-1, 1)
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

plt.savefig("/D/MIT-WHOI/github_repos/syn_model/results/n_" + str(ee).zfill(3) + "_gpsyn_single.jpg")
plt.close()
