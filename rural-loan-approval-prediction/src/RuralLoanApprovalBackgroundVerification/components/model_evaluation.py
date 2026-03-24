import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import warnings as warn 
warn.filterwarnings('ignore')
import pickle as pkl
from src.RuralLoanApprovalBackgroundVerification.utils.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from src.RuralLoanApprovalBackgroundVerification.logger import logging
from src.RuralLoanApprovalBackgroundVerification.exception import customexception
from src.RuralLoanApprovalBackgroundVerification.utils.utils import *
import mlflow 
mlflow.autolog()

class ModelEvaluation:
    def __init__(self)->None:
        pass
    def initiate_model_evaluation(self,train_array:np.array,test_array:np.array)->None:
        try:
             X_test,y_test=(test_array[:,:-1], test_array[:,-1])
             model_path=os.path.join("artifacts","model.pkl")
             model=load_object(model_path)
             #mlflow.set_registry_uri("")
             tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            #  print(tracking_url_type_store)
             with mlflow.start_run():
                y_pred=model.predict(X_test)
                mae,mse,rmse,r2,error_rate = get_metrics(y_test,y_pred)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse",mse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("error_rate", error_rate)
                 # Model registry does not work with file store
                if tracking_url_type_store != "file":
                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            raise customexception(e,sys)