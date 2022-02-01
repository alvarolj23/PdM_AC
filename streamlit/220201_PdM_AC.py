# Streamline
import streamlit as st

# Basic
import pandas as pd
from io import StringIO
from time import time
import numpy as np

# Gradient Boosting models
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier

#Light
import lightgbm as lgb
from lightgbm import LGBMClassifier


# CastBoos models
from catboost import CatBoostRegressor


# Created modules
from featuring import get_features, model_dummies, relocate_columns, decompress_pickle
from text_format import upload_success, awaiting_csv

#Images
from PIL import Image

#ColumnTransformation
from pickle import load


# Instructions for the app
st.write("""
# Compressor Predictive Maintenance

This app predicts if any component will fail in the following **2 days**!

1. Upload a CSV file with the compressor information.
2. You will get a green circle for the healthy components.
3. Red circle means a component is expected to be failing in the following 48 hours.
4. The prediction for the model, the closer to 100, means more probability for the failure to happen.

""")

# Create the header for the CSV
st.sidebar.header('Compressor Status CSV')

# Example of CSV
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

days_advance = st.sidebar.slider('Days in Advance Detection', 1, 21, 10)


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Conditioning the raw data used for the generating the model
compressors_all_data_raw = pd.read_csv('../data/compressors_all_data_hourly_raw.csv')
compressors_all_data_raw = compressors_all_data_raw.drop(
    columns=['comp1_fail', 'comp2_fail', 'comp3_fail', 'comp4_fail'])
compressors_all_data_raw['datetime'] = pd.to_datetime(compressors_all_data_raw['datetime'], format="%Y-%m-%d %H:%M:%S")

# Create lists with names of the columns for easier selection
cols_sensors_model = ['current', 'rpm', 'pressure', 'vibration']  # streamlit -> checkbox selection
cols_errors_model = ['error1', 'error2', 'error3', 'error4', 'error5']
cols_maint_model = ['comp1_maint', 'comp2_maint', 'comp3_maint', 'comp4_maint']
cols_comp_info_model = ['age', 'model']
cols_failures_model = ['comp1_fail', 'comp2_fail', 'comp3_fail', 'comp4_fail']
cols_seasonality = ['s-24', 'c-24', 's-168', 'c-168', 's-8760', 'c-8760']

# Creates the times to be used
time_lagged_sensors = 24
time_errors_count = 48
time_window_resample_hours = 3
time_advanced_failure_detection = 24 * 2

# Apply featuring engineering
compressors_all_data_raw = get_features(compressors_all_data_raw, time_window_resample_hours, cols_sensors_model,
                                        time_lagged_sensors, cols_comp_info_model, time_errors_count, cols_errors_model,
                                        cols_maint_model)

# Load model
#load_clf_comp1 = lgb.LGBMClassifier()
load_clf_comp1=lgb.Booster(model_file="../models/lgb_model_comp1_fail.model")
#load_clf_comp2 = lgb.LGBMClassifier()
load_clf_comp2=lgb.Booster(model_file="../models/lgb_model_comp2_fail.model")
#load_clf_comp3 = lgb.LGBMClassifier()
load_clf_comp3=lgb.Booster(model_file="../models/lgb_model_comp3_fail.model")
#load_clf_comp4 = lgb.LGBMClassifier()
load_clf_comp4=lgb.Booster(model_file="../models/lgb_model_comp4_fail.model")

# XGB
#load_clf_comp1 = lgb.LGBMClassifier()
#load_clf_comp1.load_model("lgb_model_comp1_fail.model")
#load_clf_comp2 = xgb.Booster()
#load_clf_comp2.load_model("lgb_model_comp2_fail.model")
#load_clf_comp3 = xgb.Booster()
#load_clf_comp3.load_model("lgb_model_comp3_fail.model")
#load_clf_comp4 = xgb.Booster()
#load_clf_comp4.load_model("lgb_model_comp4_fail.model")

#Load Regression model
load_reg_comp1 = CatBoostRegressor()
load_reg_comp1.load_model("../models/cb_model_rul_comp1.model")
load_reg_comp2 = CatBoostRegressor()
load_reg_comp2.load_model("../models/cb_model_rul_comp2.model")
load_reg_comp3 = CatBoostRegressor()
load_reg_comp3.load_model("../models/cb_model_rul_comp3.model")
load_reg_comp4 = CatBoostRegressor()
load_reg_comp4.load_model("../models/cb_model_rul_comp4.model")


if uploaded_file is not None:

    # Show a message CSV has been loaded
    upload_success('CSV has been correctly loaded')

    # Start tracking the time
    start_time = time()

    # Can be used wherever a "file-like" object is accepted:
    input_df = pd.read_csv(uploaded_file)

    
    #Identification of the model
    model_comp =input_df['model'][0]
    year_comp = input_df['age'][0]

    #COMPRESSOR BEING ANALISED
    st.title('Compressor being analised')

    # Create table
    co1, co2 = st.beta_columns([3,2])
    
    #Message with picture
    co1.image(Image.open('Pictures/Comp1.png'), use_column_width=False)
    co2.write(f"Model of the compressor: {model_comp}")
    co2.write(f"Years of the compressor: {year_comp}")


    # Conditioning of input data
    # the input might not have data for the whole year
    input_df.dropna(axis=0, how='all', inplace=True)

    # Pass datetime format
    input_df['datetime'] = pd.to_datetime(input_df['datetime'], format="%Y-%m-%d %H:%M:%S")

    # Apply featuring engineering
    input_df = get_features(input_df, time_window_resample_hours, cols_sensors_model, time_lagged_sensors,
                            cols_comp_info_model, time_errors_count, cols_errors_model, cols_maint_model)

    # Combines user input features with entire penguins dataset
    # Concat to train the model
    df = pd.concat([input_df, compressors_all_data_raw], axis=0)

    # Model the dummies
    df = model_dummies(df)

    # Relocate columns
    df = relocate_columns(df, cols_seasonality)

    #Load Yeo-Jhonson model
    yj = load(open('../models/yj.pkl', 'rb'))

    #Apply Yeo-Jhonson
    df_yj = yj.transform(df)
    df_yj = pd.DataFrame(df_yj, columns=df.columns)

    # Select only input data
    df = df[:int(len(input_df))]  # Selects only the first row (the user input data)
    df_yj = df_yj[:int(len(input_df))] # Selects only the first row (the user input data)
 
    #Drop the max and min
    #df_dm = df.drop(['current_max_lag_24h', 'rpm_max_lag_24h', 'pressure_max_lag_24h','vibration_max_lag_24h',
                    #'current_min_lag_24h', 'rpm_min_lag_24h', 'pressure_min_lag_24h','vibration_min_lag_24h'], axis =1)

    # Transform to DMatrix
    #df_dm = xgb.DMatrix(df_dm.values) #XGB
    df_dm= df #LGB

    # Apply model to make predictions
    prediction_comp1_cl = load_clf_comp1.predict(df_dm)
    prediction_comp1_rg = load_reg_comp1.predict(df_yj.iloc[-1])

    prediction_comp2_cl = load_clf_comp2.predict(df_dm)
    prediction_comp2_rg = load_reg_comp2.predict(df_yj.iloc[-1])

    prediction_comp3_cl = load_clf_comp3.predict(df_dm)
    prediction_comp3_rg = load_reg_comp3.predict(df_yj.iloc[-1])

    prediction_comp4_cl = load_clf_comp4.predict(df_dm)
    prediction_comp4_rg = load_reg_comp4.predict(df_yj.iloc[-1])

    st.title('Component failures')

    # Create table
    col1, col2, col3, col4 = st.beta_columns(4)
    RUL_critical = days_advance #days to consider the RUL as critical
    replacement_text = '<p style="font-family:Courier; color:Red; font-size: 20px;">Needs urgent replacement</p>'

    #Component 1
    col1.header("Comp 1")
    if prediction_comp1_cl[-1] < 0.5:
        col1.image('Pictures/green_circle.png', use_column_width=True)
    else:
        col1.image('Pictures/red_circle.png', use_column_width=True)
    col1.subheader('Prediction')
    col1.write(f"{(prediction_comp1_cl[-1] * 100):.2f} %")
    col1.subheader('RUL (Days)')
    if prediction_comp1_rg < RUL_critical:
        col1.write(str("{:.2f}".format(prediction_comp1_rg)))
        col1.write(replacement_text, unsafe_allow_html=True)
    else:
        col1.write(str("{:.2f}".format(prediction_comp1_rg)))

    #Component 2
    col2.header("Comp 2")
    if prediction_comp2_cl[-1] < 0.5:
        col2.image('Pictures/green_circle.png', use_column_width=True)
    else:
        col2.image('Pictures/red_circle.png', use_column_width=True)
    col2.subheader('Prediction')
    col2.write(f"{(prediction_comp2_cl[-1] * 100):.2f} %")
    col2.subheader('RUL (Days)')
    if prediction_comp2_rg < RUL_critical:
        col2.write(str("{:.2f}".format(prediction_comp2_rg)))
        col2.write(replacement_text, unsafe_allow_html=True)
    else:
        col2.write(str("{:.2f}".format(prediction_comp2_rg)))

    #Component 3
    col3.header("Comp 3")
    if prediction_comp3_cl[-1] < 0.5:
        col3.image('Pictures/green_circle.png', use_column_width=True)
    else:
        col3.image('Pictures/red_circle.png', use_column_width=True)
    col3.subheader('Prediction')
    col3.write(f"{(prediction_comp3_cl[-1] * 100):.2f} %")
    col3.subheader('RUL (Days)')
    if prediction_comp3_rg < RUL_critical:
        col3.write(str("{:.2f}".format(prediction_comp3_rg)))
        col3.write(replacement_text, unsafe_allow_html=True)
    else:
        col3.write(str("{:.2f}".format(prediction_comp3_rg)))

    #Component 4
    col4.header("Comp 4")
    if prediction_comp4_cl[-1] < 0.5:
        col4.image('Pictures/green_circle.png', use_column_width=True)
    else:
        col4.image('Pictures/red_circle.png', use_column_width=True)
    col4.subheader('Prediction')
    col4.write(f"{(prediction_comp4_cl[-1] * 100):.2f} %")
    col4.subheader('RUL (Days)')
    if prediction_comp4_rg < RUL_critical:
        col4.write(str("{:.2f}".format(prediction_comp4_rg)))
        col4.write(replacement_text, unsafe_allow_html=True)

    else:
        col4.write(str("{:.2f}".format(prediction_comp4_rg)))


    # Final time
    final_time = time() - start_time
    st.write(f'Total time for running the models: {final_time:.2f} seconds')
else:

    awaiting_csv("Awaiting CSV file to be uploaded. Please upload a CSV file in the left panel")
    input_df = pd.DataFrame()
