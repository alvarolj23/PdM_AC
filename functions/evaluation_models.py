#Regular expressions
import pandas as pd

# Utilities
from time import time

# Classification
from sklearn.model_selection import cross_validate

# Metrics
from sklearn.metrics import  recall_score, accuracy_score, precision_score, f1_score , roc_auc_score
from sklearn.metrics import make_scorer


# Define dictionary with performance metrics
scoring = {'accuracy':make_scorer(accuracy_score), 
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score), 
           'f1_score':make_scorer(f1_score),
           'auc_score':make_scorer(roc_auc_score)
           }


# Define the models evaluation function
def models_evaluation(X, y, folds, name, model):
    
    '''
    Function that runs the cross-validation (CV) for the named algorithm
    X : data set features
    y : data set target
    folds : number of cross-validation folds
    name: Designation of the algorithm. 
    model: Algorithm to be applied.
    
    '''

    # set start time
    start_time = time()

    # Perform cross-validation to each machine learning classifier
    temp_model = cross_validate(model, X, y, cv=folds, scoring=scoring)

    # set end time: execution time
    exec_time = time() - start_time
    
    #ratio score/time
    auc_time = (temp_model['test_auc_score'].mean())/exec_time
    

    # Create a data frame with the models perfoamnce metrics scores
    models_scores_table = pd.DataFrame({name:[temp_model['test_accuracy'].mean(),
                                          temp_model['test_precision'].mean(),
                                          temp_model['test_recall'].mean(),
                                          temp_model['test_f1_score'].mean(),
                                          temp_model['test_auc_score'].mean(),
                                          exec_time,
                                          auc_time]},
                                    index=['Accuracy', 'Precision', 'Recall', 'F1 Score' , 'AUC Score' , 'Exec Time', 'AUC/time'])
    

    
    # Return models performance metrics scores data frame
    return(models_scores_table)