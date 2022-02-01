# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
# Data 
import numpy as np
import pandas as pd

#Regular expressions
import re as re

# Utilities
from time import time

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale


from sklearn import model_selection


# Metrics
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import make_scorer


# Others
import warnings
from functools import reduce
from pathlib import Path
import os

#Regression metrics
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, explained_variance_score as evs, mean_squared_log_error as msle





def run_cv_reg(models, features, target, splits,evaluator='r2'):
    '''
    Function that runs the cross-validation (CV) for the named algorithm
    
    Parameters:
    * models = List of Tuples: (name, model). Algorithms to be applied.
    * features = matrix with the predictors
    * target = matrix with the targets
    * evaluator = metric used to evaluate the model (predefined: r2_score).
    '''
    # Set seed to obtain the same random numbers
    seed = 7

    # Evaluate each model
    names = []
    component = []
    results = []
    mins = []
    quartiles_1 = []
    medians = []
    means = []
    stds = []
    quartiles_3 = []
    maxs = []
    times = []
    scoring = evaluator
    
    # Executing the function for every model in the list: models
    for name, model in models:
        # set start time
        print(f'Executing {name}...')
        start_time = time()

        for rul_comp in target:

          # Missing Data ---> dropNa
            # First for the rul_comp
          y_temp = target[rul_comp].dropna()
          X_temp = features.loc[list(y_temp.index)]
            # Second for the variables
          X_temp = X_temp.dropna()
          y_temp = y_temp.loc[list(X_temp.index)]

          
          kfold = model_selection.KFold(n_splits=splits, random_state=seed, shuffle=True)
          cv_results = model_selection.cross_val_score(estimator=model,
                                                     X=X_temp,
                                                     y=y_temp,
                                                     cv=kfold,
                                                     scoring=scoring)
          # appending stats to lists
          names.append(name)
          component.append(rul_comp)
          results.append(cv_results)
          mins.append(cv_results.min())
          quartiles_1.append(np.percentile(cv_results, 25)) # Q1
          medians.append(np.median(cv_results)) # Q2 = median
          means.append(cv_results.mean())
          stds.append(cv_results.std())
          quartiles_3.append(np.percentile(cv_results, 75)) # Q3
          maxs.append(cv_results.max())

          # set end time: execution time
          exec_time = time() - start_time

          # Appending to the main list
          times.append(exec_time)
          print(f'CV finished for {name} and {rul_comp}')
        
    # Creating a DataFrame to see the performance of each model:
    df_models = pd.DataFrame({'model': names,
                              'component': component,
                              'min_r2_score': mins,
                              '1st_quantile': quartiles_1,
                              'median_r2_score': medians,
                              'mean_r2_score': means,
                              'std_r2_score': stds,
                              '3rd_quantile': quartiles_3,
                              'max_r2_score': maxs,
                              'exec_time_sec': times})
    
    # Creating a time_score_ratio
    df_models['time_score_ratio'] = df_models['mean_r2_score'] / df_models['exec_time_sec']

    
    # Rounding to 4 decimals
    round_cols = dict(zip(df_models.columns, len(df_models.columns)*[4]))
    df_models = df_models.round(round_cols)
    
    return (df_models, results)


    
    
