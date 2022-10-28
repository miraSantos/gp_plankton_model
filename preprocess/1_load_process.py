import pandas as pd
from calendar import month_name as mn
import datetime as datetime
import numpy as np

months = mn[1:]  # importing month names from calendar


def add_date_columns(df, date_column, date_format):
    df["date_time"] = pd.to_datetime(df[date_column],
                                     format=date_format)  # reconverting to date column to datetime format
    df["date"] = df["date_time"].dt.date
    df["month"] = df["date_time"].dt.month  # extracting month
    df["doy"] = pd.to_datetime(df["date_time"].dt.dayofyear, format="%j")  # extracting day of year
    df["doy_numeric"] = df["date_time"].dt.dayofyear  # extracting day of year
    df["year"] = df["date_time"].dt.year  # extracting month
    df["monthname"] = pd.Categorical(df["date_time"].dt.month_name(), categories=months, ordered=True)
    df = df.dropna()
    df.set_index("date", inplace=True)
    return df


if __name__ == '__main__':

    # IMPORTING SYN
    daily_time = pd.read_csv("/dos/MIT-WHOI/github_repos/syn_model/data/dailysyntime_matrix.txt", sep="\t", header=None,
                             names=["datetime"])
    daily_syn = pd.read_csv("/dos/MIT-WHOI/github_repos/syn_model/data/dailysynconc_matrix.txt", sep="\t", header=None,
                            names=["synconc"])
    df_syn_daily = pd.concat([daily_time, daily_syn], axis=1)

    # IMPORTING ENV DATA
    df_env_daily = pd.read_csv("/dos/MIT-WHOI/github_repos/syn_model/data/mvco_daily.csv")

    # IMPORTING NUTRIENT DATA
    heads = ["Event_Num", "Event_Num_Niskin", "Start_Date", "Start_Time_UTC", "Lat", "Lon", "Depth", "NO3_a", "NO3_b",
             "NO3_c", "NH4_a", "NH4_b", "NH4_c", "SiO2_a", "SiO2_b", "SiO2_c", "PO4_a", "PO4_b", "PO4_c"]
    df_nut = pd.read_csv("/dos/MIT-WHOI/github_repos/syn_model/data/mvco_nutrients.csv")
    df_nut.columns = heads
    df_nut["date"] = pd.to_datetime(df_nut["Start_Date"],
                                    format='%Y-%m-%d %H:%M:%S.%f').dt.date
    df_nut["time"] = pd.to_datetime(df_nut["Start_Time_UTC"],
                                    format='%Y-%m-%d %H:%M:%S.%f').dt.time
    df_nut["datetime"] = df_nut.apply(lambda r: datetime.datetime.combine(r['date'], r['time']), 1)

    # importing Concentrations of other IFCB classes
    df_conc = pd.read_csv("/dos/MIT-WHOI/data/2021/concentration_by_class_time_series_CNN_daily08Sep2021.csv")

    # adding date columns
    df_syn_daily = add_date_columns(df_syn_daily, date_column="datetime", date_format='%Y-%m-%d')
    df_env_daily = add_date_columns(df_env_daily, date_column="days", date_format='%d-%b-%Y')
    df_nut = add_date_columns(df_nut, "datetime", '%Y-%m-%d %H:%M:%S.%f')
    df_conc = add_date_columns(df_conc, date_column="datetime", date_format="%d-%b-%Y %H:%M:%S")

    df_merged = df_env_daily.combine_first(df_syn_daily).combine_first(df_nut).combine_first(df_conc)
    df_merged = df_merged.replace(r'^\s*$', np.nan, regex=True)

    historical_mean = df_merged.groupby("doy_numeric").mean()["synconc"]

    index_list = ~(df_merged.synconc.isna())
    df_merged["synconc_subtracted"] = np.nan
    df_merged.loc[df_merged.index[index_list], "synconc_subtracted"] = df_merged.loc[df_merged.index[index_list],"synconc"].values - historical_mean[df_merged.loc[df_merged.index[index_list], "doy_numeric"]].values

    save_path = "/home/mira/PycharmProjects/gp_plankton_model/datasets/df_merged_daily_phyto_"+str(datetime.date.today())
    df_merged.to_pickle(save_path+".pkl")
    df_merged.to_csv(save_path+".csv",encoding='utf-8')
