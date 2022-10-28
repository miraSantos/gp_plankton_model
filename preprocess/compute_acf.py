import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf #compute autocorrelation
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns



## READING IN DATA

path = "/dos/MIT-WHOI/data/2022/dfmerged_daily_phyto_2022-09-06.pkl"

df = pd.read_pickle(path)
df = df[~(df.index == 0)]
df.head()

#computing historical mean
historical_mean = df.groupby("doy_numeric").mean()["synconc"]

#subtracting the historical mean and adding new column to dataframe
index_list = ~(df.synconc.isna())
df["synconc_subtracted"] = np.nan
df.loc[df.index[index_list],"synconc_subtracted"] = df.loc[df.index[index_list],"synconc"].values - historical_mean[df.loc[df.index[index_list],"doy_numeric"]].values


df.to_pickle("/dos/MIT-WHOI/data/2022/dfmerged_daily_phyto_"+"2022_Oct"+".pkl")
df.to_csv("/dos/MIT-WHOI/data/2022/dfmerged_daily_phyto_"+"2022_Oct"+".csv")


def autocorrelate(dfcol, nlag, ac_threshold):
    # NUMBER OF LAGS

    # calcuate autocorrelation factor for ssyn values
    acfout = acf(dfcol, nlags=nlag, missing='conservative')

    acfseries = pd.Series(acfout)

    ac_threshold = 0.7
    acfseries[acfseries >= ac_threshold].idxmin()
    max_days = acfseries[acfseries >= ac_threshold].idxmin()
    print("Max number of days (where AC >= threshold): " + str(max_days) + " days")

    x = np.linspace(1, nlag + 1, nlag + 1)

    # FIGURE SIZE
    plt.figure(figsize=(15, 8), dpi=500)
    # SCALE and GRID STYLE
    sns.set(font_scale=2, style="whitegrid")
    # FONT
    #     plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.serif": ["Computer Modern Roman"]})

    plt.stem(x, acfout)
    # plt.xlim([0,100])
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title(str(dfcol.name) + " autocorrelation over " + str(nlag) + " days")

    # FIGURE SIZE
    plt.figure(figsize=(15, 8), dpi=500)
    # SCALE and GRID STYLE
    #     sns.set(font_scale = 2, style = "whitegrid")
    # FONT
    plt.stem(x, acfout)
    plt.xlim([0, 100])

    # plotting labels to show points above threshold
    plt.stem(x[:max_days], acfout[:max_days], linefmt='C1')
    plt.axvline(x=max_days, color="r", linestyle='dashed')
    plt.axhline(y=acfout[max_days - 1], color='r', linestyle='dashed')
    plt.text(max_days + 1, 1, str(max_days) + " days", color="r")
    # plt.xticks(np.arange(0, len(x)+1, 10))

    # plt.xticks(np.arange(0, len(x)+1, 10))

    plt.xlabel("Lag (days)")
    plt.ylabel("Autocorrelation")
    plt.title(str(dfcol.name) + " autocorrelation over 100 days")


autocorrelate(df.synconc, 2000, 0.7)