#mira santos

import pandas as pd

from calendar import month_name as mn
import datetime
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.dates as mdates #for working with dates in plots

import argparse
from os.path import exists
from urllib.request import urlopen

import numpy as np
import torch

import pyro
from pyro.contrib.timeseries import IndependentMaternGP, LinearlyCoupledMaternGP

def download_data(data_path):
    """
    :param data_path(str): path to data
    :return:
    """
    df = pd.read_csv(data_path) # .drop_duplicates(subset = "date") #remove duplicate days

    months = mn[1:]  # importing month names from calendar

    # reformat dates to datetime format
    df.date = pd.to_datetime(df.date, format="%Y-%m-%d")
    df.year = df.date.dt.year
    df.month = df.date.dt.month
    df["doy_numeric"] = df.date.dt.dayofyear
    df["doy"] = pd.to_datetime(df.date.dt.dayofyear, format="%j")
    df["monthname"] = pd.Categorical(df.date.dt.month_name(), categories=months, ordered=True)
    df["year_numeric"] = df.year + df.doy_numeric / 365

    df["temp"] = df.Beam_temperature_corrected
    return df

def create_daily_df(df):
    """
    :param df(pandas.dataframe): dataframe
    :return: df with daily frequency
    """
    #SETTING DAILY FREQUENCY TO DAT
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
    dfd["year"] = dfd.date.dt.year
    dfd["log_syn"] = np.log(dfd.synconc)

    dfd.head()
    return dfd

def interpolate(dataframe,column):
    """
    :param dataframe(pandas.dataframe): dataframe with data
    :param column (pandas data column):
    :return: dataframe(pandas.dataframe): augmented dataframe with interpolatio
    """
    #INTERPOLATION
    print("number of NaNs in " + str(column) + ":" + str(dataframe[column].isna().sum()))
    dataframe[column+"_filled"] = dataframe[column].interpolate(method = "cubic",limit = 7,limit_area = "inside")
    dataframe[column+"_interpolated"] = dataframe.groupby("doy_numeric")[column+"_filled"].apply(lambda x: x.fillna(x.mean())) #calculating the mean
    print("number of NaNs in " + str(column)  + ":" + " after interpolation:" + str(dataframe[column+"_interpolated"].isna().sum()))

    #INTERPOLATING THE STANDARD DEVIATION

    default_std = 0.1    #entries with raw observations have this standard deviation

    dataframe.loc[:,column+"_std"] = dataframe[column].values
    dataframe[column+"_std"] = dataframe.groupby("doy_numeric")[column+"_std"].apply(lambda x: x.fillna(x.std())) #calculating the standard deviation and filling in those values
    idx_notnull = np.where(dataframe[column].notnull())[0] #retrieve index of non zero entries in column
    dataframe.loc[idx_notnull,[column+"_std"]] = default_std #assign
    dataframe["daily_index"] = dataframe.index
    return dataframe


if __name__ == '__main__':
    data_path =  "/dos/MIT-WHOI/github_repos/syn_model/data/dfmerged_dailysynenv.csv"
    save_path = "/home/mira/PycharmProjects/gp_plankton_model/datasets/"
    df = download_data(data_path)
    dfd = create_daily_df(df)
    dfd = interpolate(dfd,"temp")
    dfd = interpolate(dfd,"log_syn")
    dfd.to_pickle(save_path+"syn_dataset.pkl")
    print("saved to datasets")
