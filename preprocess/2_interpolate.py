import pandas as pd
import numpy as np
import datetime

def create_daily_df(df):
    """
    :param df(pandas.dataframe): dataframe
    :return: df with daily frequency
    """
    #SETTING DAILY FREQUENCY TO DATAFRAME
    df.reset_index(inplace=True)
    dfd = df.drop_duplicates(subset = "date").copy()

    print(dfd.shape)

    # dfindexed = dfd.groupby(pd.PeriodIndex(preprocess = dfd.date,freq = "D"))
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
    :param dataframe(pandas.dataframe): dataframe with preprocess
    :param column (pandas preprocess column):
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
    data_full_path = "/dos/MIT-WHOI/data/2022/dfmerged_daily_phyto_2022-09-06.pkl"
    df = pd.read_pickle(data_full_path)
    save_path = "/dos/MIT-WHOI/data/2022/df_merged_daily_phyto_" +str(datetime.date.today()) + "_interpolated"
    dfd = create_daily_df(df)
    dfd = interpolate(dfd, "Beam_temperature_corrected")
    dfd = interpolate(dfd, "synconc")
    dfd.reset_index(inplace=True)
    dfd = dfd.replace(r'^\s*$', np.nan, regex=True)
    dfd.to_pickle(save_path+".pkl")
    print(dfd.dtypes)
    dfd.to_csv(save_path+".csv")

    print("saved to datasets")
