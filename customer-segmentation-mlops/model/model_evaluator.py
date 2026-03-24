import logging
from abc import ABC,abstractmethod
import pandas as pd
import numpy as np 
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

class ModelEvaluating(ABC):
    @abstractmethod
    def calculate_score(self,y_test:np.ndarray,y_pred:np.ndarray):
        pass 

class MeanSquaredError(ModelEvaluating):
    try: 
        def calculate_score(self,y_test:np.ndarray,y_pred:np.ndarray):
            mse = mean_squared_error(y_test,y_pred)
            return mse
    except Exception as e:
        logging.error(e)
        raise e

class MeanAbsoluteError(ModelEvaluating):
    try: 
        def calculate_score(self,y_test:np.ndarray,y_pred:np.ndarray)->float:
            mae = mean_absolute_error(y_test,y_pred)
            return mae
    except Exception as e:
        logging.error(e)
        raise e

class RootMeanSquaredError(ModelEvaluating):
    try: 
        def calculate_score(self,y_test:np.ndarray,y_pred:np.ndarray)->float:
            mse = mean_squared_error(y_test,y_pred,squared=False)
            return mse
    except Exception as e:
        logging.error(e)
        raise e

class R2Score(ModelEvaluating):
    try:
        def calculate_score(self,y_test:np.ndarray,y_pred:np.ndarray)->float:
            r2 = r2_score(y_test,y_pred)
            return r2
    except Exception as e:
        logging.error(e)
        raise e


    
