import os
import re
import pandas as pd

def clean_df_tmp(df_tmp):
    """
    Function for clean dataframe by cleaning condition.
    """
    unit_type = df_tmp['unit_type'].isin([6, 7])

    df_tmp = df_tmp[unit_type].reset_index(drop=True)
    return df_tmp

def load_gps_data(gps_data_dir):
    """
    Function for load and merge dataframes.
    """
    years_list = ['2019']
    months_list = ['03']
    days_list = ['01']
    df_list = []

    for year in years_list:
        for month in months_list:
            # get list filename in the folder path.
            file_names = os.listdir(gps_data_dir + f"/{year}-{month}/")
            for day in days_list:
                amount_day = 0
                for file_name in file_names:
                    file_name_part = re.split('-|_', file_name)
                    if len(file_name_part) > 2:
                        # select the day we chose from filename.
                        if file_name_part[2] == day:
                            _tmp = pd.read_csv(gps_data_dir + f"/{year}-{month}/" + file_name, compression='zip', parse_dates=['time_stamp'])
                            _tmp = clean_df_tmp(_tmp)
                            df_list.append(_tmp)
                            amount_day += 1
                            del _tmp

                    # the data collected was split into 8 sessions per day.
                    if amount_day > 7:
                        break
    
    df_gps = pd.concat(df_list, axis='rows', ignore_index=True)
    return df_gps

def get_road_data(road_data_dir,rd):
    df_road = pd.read_csv(road_data_dir + '/roaddb.csv')
    df_road = df_road[df_road['rd'] == rd].reset_index(drop=True)
    return df_road