from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel
from sklearn import preprocessing

from scipy import interpolate

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy

from calendar import month_name as mn
import datetime
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.dates as mdates #for working with dates in plots

############################
#Importing Data
#############################

def import_data():
    PATH = "../data/dfmerged_dailysynenv.csv"
    df = pd.read_csv(PATH)#.drop_duplicates(subset = "date") #remove duplicate days

    months = mn[1:] #importing month names from calendar

    reformat dates to datetime format
    df.date = pd.to_datetime(df.date, format = "%Y-%m-%d")
    df.year = df.date.dt.year
    df.month = df.date.dt.month
    df["doy_numeric"] = df.date.dt.dayofyear
    df["doy"] = pd.to_datetime(df.date.dt.dayofyear, format = "%j")
    df["monthname"] = pd.Categorical(df.date.dt.month_name(), categories=months, ordered = True)
    df["year_numeric"] = df.year + df.doy_numeric/365

    shortening the column name for temperature
    df["temp"] = df.Beam_temperature_corrected
    return df

############################
#SETTING TO DAILY FREQUENCY
#############################
#SETTING DAILY FREQUENCY TO DATA
dfd = df.drop_duplicates(subset = "date").copy()

print(dfd.shape)

# dfindexed = dfd.groupby(pd.PeriodIndex(data = dfd.date,freq = "D"))
dfd = dfd.set_index("date",inplace = False)
dfd.head()

dfd =dfd.asfreq("D")
dfd = dfd.reset_index()

print("shape after setting daily frequency: " + str(dfd.shape))
dfd["lindex"] = dfd.index

dfd["doy_numeric"] = dfd.date.dt.dayofyear

dfd.head()

############################
#INTERPOLATION
#############################

#INTERPOLATING TEMPERATURE
print("number of NaNs in temp: " + str(dfd["temp"].isna().sum()))
dfd["temp_interpolated"] = dfd.groupby("doy_numeric")["temp"].apply(lambda x: x.fillna(x.mean())) #calculating the mean
print("number of NaNs in temp after interpolation: " + str(dfd["temp_interpolated"].isna().sum()))

#INTERPOLATING THE STANDARD DEVIATION
dfd["temp_std"] = dfd.temp
dfd["temp_std"] = dfd.groupby("doy_numeric")["temp_std"].apply(lambda x: x.fillna(x.std())) #calculating the mean
dfd.temp_std.loc[np.where(dfd['temp'].notnull())[0]] = 0

#INTERPOLATING SYN
dfd["log_syn"] = np.log(dfd.synconc)

print("number of NaNs in syn: " + str(dfd["log_syn"].isna().sum()))
dfd["log_syn_interpolated"] = dfd.groupby("doy_numeric")["log_syn"].apply(lambda x: x.fillna(x.mean())) #calculating the mean
print("number of NaNs in syn after interpolation: " + str(dfd["log_syn_interpolated"].isna().sum()))


#INTERPOLATING THE STANDARD DEVIATION
dfd["log_syn_std"] = dfd.log_syn
dfd["log_syn_std"] = dfd.groupby("doy_numeric")["log_syn_std"].apply(lambda x: x.fillna(x.std())) #calculating the mean
dfd["log_syn_std"].loc[np.where(dfd['log_syn'].notnull())[0]] = 0

################################
#Preparing Data for the Model
################################

#Preparing the data for GP fit

dfd["Year_sin"] = 4*np.sin(2*np.pi/365*dfd.daily_index)
dfd["Year_cos"] = 4*np.cos(2*np.pi/365*dfd.daily_index)

X_temp = dfd[["Year_sin","Year_cos","temp_interpolated"]].to_numpy()
y_temp = dfd[["log_syn_interpolated"]].to_numpy()
y_mean_temp = y_temp.mean()
y_temp = y_temp - y_mean_temp

plt.plot(y_temp)

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

#CREATING GPR MODEL
gpr_temp = GaussianProcessRegressor(kernel = final_kernel , alpha = 1e-5,n_restarts_optimizer=10)

end = 2/3*len(y)

for ee in range(2,end):     
    #FITTING DATA TO THE MODEL
    gpr_temp.fit(X_temp[aa:ee],y_temp[aa:ee])


    today = datetime.datetime.now()
    current_day = today.year + today.day / 365
    X_test = np.linspace(start=2003, stop = 2018, num=1_000).reshape(-1, 1)
    mean_y_pred, std_y_pred = gaussian_process.predict(X_test, return_std=True)
    mean_y_pred += y_mean

    # mean_y_pred = np.exp(mean_y_pred)
    # std_y_pred = np.exp(std_y_pred)

    plt.figure(figsize=(15, 8))
    
    plt.fill_between(
    X_test.flatten()[1:ee],
    (mean_y_pred - 3*std_y_pred).flatten()[1:ee],
    (mean_y_pred + 3*std_y_pred).flatten()[1:ee],
    color="tab:blue",
    alpha=0.043,
    label = "+- 3 std"
    )

    plt.fill_between(
    X_test.flatten()[1:ee],
    (mean_y_pred - 2*std_y_pred).flatten()[1:ee],
    (mean_y_pred + 2*std_y_pred).flatten()[1:ee],
    color="tab:blue",
    alpha=0.27,
    label = "+- 2 std"
    )


    plt.fill_between(
    X_test.flatten()[1:ee],
    (mean_y_pred - std_y_pred).flatten()[1:ee],
    (mean_y_pred + std_y_pred).flatten()[1:ee],
    color="tab:blue",
    alpha=0.68,
    label = "+- 1 std"
    )  

    
    #plt.scatter(X[1:ee], y[1:ee],color="black",marker = "x", label="Measurements")
    #plt.plot(X_test, mean_y_pred, color="tab:blue", alpha=0.4, label="Gaussian process")
    #plt.fill_between(
        #X_test.ravel(),
        #mean_y_pred - std_y_pred,
        #mean_y_pred + std_y_pred,
        #color="tab:blue",
        #alpha=0.2,
    #)
    #plt.ylim([0,13])
    #plt.legend()
    #plt.xlabel("Year")
    #plt.ylabel("Log(Daily Average of Syn Concentration)")
    #_ = plt.title(
        #"Daily Average of Syn Concentration"
    #)
    
    plt.savefig("/D/MIT-WHOI/github_repos/syn_model/results/gp_syn_plots_temp/" + str(ee).zfill(3) + "gpsyn.jpg")
    print(str(ee).zfill(3) + " of " + str(end))
    plt.close()
