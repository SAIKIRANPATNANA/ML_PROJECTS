import os
import sys
import pickle as pkl
import numpy as np
import pandas as pd
from src.RuralLoanApprovalBackgroundVerification.logger import logging
from src.RuralLoanApprovalBackgroundVerification.exception import customexception
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pkl.dump(obj, file_obj)
    except Exception as e:
        logging.info('Exception occured in saving model.')
        raise customexception(e, sys)
def get_metrics(y_test,y_pred):
    y_pred_df = pd.DataFrame(y_pred,columns=['y_pred'])
    y_test_df = pd.DataFrame(y_test,columns=['y_test'])
    result = pd.concat([y_test_df, y_pred_df],axis=1)
    result['error_rate'] = abs((result['y_test']-result['y_pred'])/result['y_test'])
    error_rate = 1-result['error_rate'].mean()
    mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test,y_pred)
    return mae,mse,rmse,r2,error_rate
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            _,_,_,_,error_rate = get_metrics(y_test,y_pred)
            report[list(models.keys())[i]] = error_rate
        return report
    except Exception as e:
        logging.info('Exception occured in model training.')
        raise customexception(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pkl.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in loading pickle file.')
        raise customexception(e,sys)

    