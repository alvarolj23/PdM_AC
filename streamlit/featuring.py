import pandas as pd
import numpy as np
import bz2
import pickle
import _pickle as compPickle

# SENSORS
def sensors_features(df,time_window_resample_hours,cols_sensors_model, time_lagged_sensors):
    """
    Takes the dataframe and returns the sensors mean value in the time window and the mean for the lagging

    :param df:
    :return: sensors_mean_tw, sensors_mean_lagging
    """
    # 3H Time-Window
    # Temporary list containing the pivot table
    temp = [
        pd.pivot_table(
            df,
            index="datetime",
            columns="compressorID",
            values=col).resample(str(time_window_resample_hours) + "H", closed="left", label="right").mean().unstack()
        for col in cols_sensors_model
    ]
    # Concat the pivot table and create a dataframe
    sensors_mean_tw = pd.concat(temp, axis=1)  # Unify the series
    sensors_mean_tw.columns = [col + "_mean_" + str(time_window_resample_hours) + "h" for col in
                               cols_sensors_model]  # Asign names to the columns
    sensors_mean_tw.reset_index(inplace=True)

    # 24H Lagged-Features
    # Create temporary list
    temp = []
    temp = [
        pd.pivot_table(
            df,
            index="datetime",
            columns="compressorID",
            values=col).rolling(window=time_lagged_sensors).mean().
            resample(str(time_window_resample_hours) + "H", closed="left", label="right").first().unstack()
        for col in cols_sensors_model
    ]
    # Concat all temporary lists
    sensors_mean_lagging = pd.concat(temp, axis=1)  # Unify the series
    sensors_mean_lagging.columns = [col + "_mean_lag_" + str(time_lagged_sensors) + "h" for col in
                                    cols_sensors_model]  # Asign names to the columns
    sensors_mean_lagging.reset_index(inplace=True)

    # For MAX
    temp = []
    temp = [
        pd.pivot_table(
            df,
            index="datetime",
            columns="compressorID",
            values=col).rolling(window=time_lagged_sensors).max().
            resample(str(time_window_resample_hours) + "H", closed="left", label="right").first().unstack()
        for col in cols_sensors_model
    ]
    sensors_max_lagging = pd.concat(temp, axis=1) # Unify the series
    sensors_max_lagging.columns = [col + "_max_lag_" + str(time_lagged_sensors) + "h" for col in 
                                   cols_sensors_model] # Asign names to the columns
    sensors_max_lagging.reset_index(inplace=True) 

    # For MIN
    temp = []
    temp = [
        pd.pivot_table(
            df,
            index="datetime",
            columns="compressorID",
            values=col).rolling(window=time_lagged_sensors).min().
            resample(str(time_window_resample_hours) + "H", closed="left", label="right").first().unstack()
        for col in cols_sensors_model
    ]
    sensors_min_lagging = pd.concat(temp, axis=1) # Unify the series
    sensors_min_lagging.columns = [col + "_min_lag_" + str(time_lagged_sensors) + "h" for col in 
                                   cols_sensors_model] # Asign names to the columns
    sensors_min_lagging.reset_index(inplace=True) 

    return sensors_mean_tw, sensors_mean_lagging , sensors_max_lagging, sensors_min_lagging


def errors_features(df,time_errors_count,time_window_resample_hours,cols_errors_model):
    """
    Takes the dataframe and returns the errors dummies and the errors count
    :param df: 
    :return: errors_dum , error_count_total
    """
    # Filter by dates

    errors_dum = pd.get_dummies(df[['datetime' , 'compressorID']+cols_errors_model]) # We put a 1 if the error appears for that machine, 0 otherwise.
    errors_dum.columns = ["datetime", "compressorID", "error1", "error2", "error3", "error4", "error5"]
    #Fill NaN errrors_dum with zero
    errors_dum.fillna(0, inplace=True)

    # ERRORS COUNT
    temp = [
          pd.pivot_table(
              errors_dum,
              index="datetime",
              columns="compressorID",
              values=col).rolling(window=time_errors_count).sum().
              resample(str(time_window_resample_hours) + "H", closed="left", label="right").first().unstack()
          for col in cols_errors_model
      ]
    # Concat all temporary lists
    error_count_total = pd.concat(temp, axis=1)
    error_count_total.columns = [i + "count" for i in cols_errors_model]
    error_count_total.reset_index(inplace=True)

    return errors_dum , error_count_total

def maint_features(df,cols_maint_model):
    """
    Takes the dataframe and returns the times since last replacement for the components
    :param df: 
    :return: time_last_main
    """

    #TIME SINCE LAST MAINTENANCE

    time_last_main = df.copy()
    time_last_main = time_last_main[['datetime' , 'compressorID']+cols_maint_model]
    for comp in cols_maint_model:

        time_last_main.loc[time_last_main[comp] < 1, comp] = None
        time_last_main.loc[-time_last_main[comp].isnull(), comp] = time_last_main.loc[-time_last_main[comp].isnull(), "datetime"]
        time_last_main[comp] = pd.to_datetime(time_last_main[comp].fillna(method="ffill"))
        time_last_main[comp] = (time_last_main["datetime"] - pd.to_datetime(time_last_main[comp])) / np.timedelta64(1, "D")
    return time_last_main


def model_seasonality(df):
    """
    Create seasonality column
    :param df:
    :return:
    """
    data_points_year = 2921
    num_comp= df['compressorID'].nunique()
    time_window_resample_hours = 3


    for period in [24, 24 * 7, 24 * 365]:

      period_resample = period/time_window_resample_hours

      sine= np.sin(2 * np.pi * np.arange(data_points_year) / period_resample)
      cosine= np.cos(2 * np.pi * np.arange(data_points_year) / period_resample)

      sine_rep = np.tile(sine, num_comp)
      cosine_rep = np.tile(cosine, num_comp)

      df[f"s-{period}"] = sine_rep[:int(len(df))]
      df[f"c-{period}"] = cosine_rep[:int(len(df))] # Selects only the first row (the user input data)


def clean_format_model(df):
    """
    Delete datetime and clean NaN
    :param df: 
    :return: 
    """
    df.drop('datetime', axis=1, inplace=True)
    df.dropna(axis=0, inplace=True)

def cycle_features(df):
    """
    Create a cycle column
    :param df: 
    :return: 
    """
    #Create a list for the cycles
    cycles=[]
    # Loop the whole dataframe by compressorID and fill for each compressorID with the cycle number
    for machine in range(0,len(df.groupby('compressorID')['datetime'].count())):
        num_cycle= 0
        for num_cycle in range (1, len(df[df['compressorID'] == df['compressorID'].unique()[machine]]) +1):
          cycles.append(num_cycle)
    # Insert column in third position
    df.insert(2, 'cycle', cycles)

def get_features(df,time_window_resample_hours,cols_sensors_model, time_lagged_sensors, cols_comp_info_model , time_errors_count, cols_errors_model, cols_maint_model):
    """
    Takes the dataframe and applies the previous featuring steps
    :param df: 
    :return: df_featured: 
    """

    #From last functions
    sensors_mean_tw, sensors_mean_lagging, sensors_max_lagging, sensors_min_lagging = sensors_features(df,time_window_resample_hours,cols_sensors_model, time_lagged_sensors)
    errors_dum, error_count_total = errors_features(df,time_errors_count,time_window_resample_hours,cols_errors_model)
    time_last_main = maint_features(df, cols_maint_model)

    # MERGE FEATURED DATA
    df_featured = sensors_mean_tw.set_index('datetime').sort_values(['datetime', 'compressorID']).reset_index()

    # Merge sensors data with errors
    for df_temp in (sensors_mean_lagging,sensors_max_lagging, sensors_min_lagging, errors_dum,error_count_total,time_last_main):
        df_temp = df_temp.set_index('datetime').sort_values(['datetime', 'compressorID']).reset_index()
        df_featured = pd.merge_asof(df_featured,
                                df_temp,
                                on="datetime",
                                by="compressorID",
                                tolerance=pd.Timedelta("30min"),
                                allow_exact_matches=True)
    #Remove duplicated
    df_featured.drop_duplicates(inplace=True)

    #reset index
    df_featured = df_featured.sort_values(['compressorID','datetime']).reset_index(drop=True)

    # Merge with compressor information
    compressor_info = df[['compressorID'] + cols_comp_info_model]
    compressor_info.drop_duplicates(inplace=True)
    compressor_info = compressor_info.reset_index(drop=True)
    df_featured= df_featured.merge(compressor_info, how = 'inner' , on = 'compressorID')

    #Add the cycle column
    cycle_features(df_featured)

    #Add seasonality columns
    model_seasonality(df_featured)

    #Clean format
    clean_format_model(df_featured)


    return df_featured

def model_dummies(df):
    """
    OneHot encoding the model column
    :param df: 
    :return: df
    """
    #Select only compressorID and model
    model_dummy = df[['compressorID', 'model' , 'age']]
    #Delete duplicates
    model_dummy.drop_duplicates(inplace=True)
    #Get dummies
    model_dummy = pd.get_dummies(model_dummy, columns= ['model'], drop_first=True)
    model_dummy = model_dummy[['compressorID','model_model2' , 'model_model3' , 'model_model4' , 'age']]
    model_dummy.columns = ["compressorID", "model2", "model3", "model4" , "age" ]
    model_dummy = model_dummy.reset_index(drop=True)
    # Drop model and age
    df.drop(['model', 'age'],axis=1, inplace = True)
    #Merge with df data
    df = pd.merge(df,model_dummy, how = 'inner', on = 'compressorID')

    return df

def relocate_columns(df, cols_seasonality):
    """
    Order dataframe to input in the model
    :param df:
    :return: df
    """
    #Relocate columns
    df_seasonality = df[cols_seasonality]
    df.drop(cols_seasonality, axis=1, inplace= True)
    df = pd.concat([df, df_seasonality], axis=1)

    return df

def decompress_pickle(file):
    '''
    Decompress a compressed pickle.
    Params:
    :file: file to be decompressed
    '''
    data = bz2.BZ2File(file, 'rb')
    data = compPickle.load(data)
    return data

