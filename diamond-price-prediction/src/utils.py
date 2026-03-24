import sys,os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import pickle as pkl
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pkl.dump(obj,file_obj)
    except Exception as e:
        logging.info('Exception occured in saving the object')
        raise CustomException(e,sys)
def load_obj(file_path):
    try:
        with open(file_path,'rb'):
            return (pkl.load(file_path))
    except Exception as e:
        logging.info('Exception Occured in loading the object')
        raise(e,sys)
def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report = dict()
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(x_train,y_train)
            y_pred_test = model.predict(x_test)
            r2_square = r2_score(y_test,y_pred_test)
            report[list(models.keys())[i]] = r2_square
        return report 
    except Exception as e:
        logging.info('Exception occured in evaluating the model')
        raise CustomException(e,sys)

        