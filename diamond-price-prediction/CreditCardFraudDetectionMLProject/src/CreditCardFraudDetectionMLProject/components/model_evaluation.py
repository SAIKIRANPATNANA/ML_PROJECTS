import os 
import sys
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from src.CreditCardFraudDetectionMLProject.logger import logging
from src.CreditCardFraudDetectionMLProject.exception import CustomException
from src.CreditCardFraudDetectionMLProject.components.data_ingestion import DataIngestion
from src.CreditCardFraudDetectionMLProject.components.data_validation import DataValidation
from src.CreditCardFraudDetectionMLProject.components.data_transformation import DataTransformation
from src.CreditCardFraudDetectionMLProject.components.data_transformation import DataTransformation
from src.CreditCardFraudDetectionMLProject.components.model_trainer import ModelTraining
from src.CreditCardFraudDetectionMLProject.utils.utils import *
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow 
mlflow.autolog()
import warnings as warn
warn.filterwarnings('ignore')

@dataclass
class ModelEvalautionConfig:
    confusion_matrix_path = os.path.join('artifacts','confusion_matrix.png')
    auc_roc_path = os.path.join('artifacts','auc_roc.png')
class ModelEvaluation:
    def __init__(self):
        self.model_evaluation_config = ModelEvalautionConfig()
    def initiate_model_evaluation(self,test_array,model_path):
        try:
            x_test = test_array[:,:-1]
            y_test = test_array[:,-1]
            model = load_object(model_path)
            y_pred = model.predict(x_test)
            accuracy, precision, recall, f1 = evaluate_metrics(y_test,y_pred)
            plot_auc_roc_curve(y_test,y_pred,self.model_evaluation_config.auc_roc_path)
            plot_confusion_matrix(y_test,y_pred,self.model_evaluation_config.confusion_matrix_path)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            with mlflow.start_run():
                mlflow.log_metric('accuracy',accuracy)
                mlflow.log_metric('precision', precision)
                mlflow.log_metric('recall',recall)
                mlflow.log_metric('f1',f1)
                mlflow.log_artifact(self.model_evaluation_config.confusion_matrix_path)
                mlflow.log_artifact(self.model_evaluation_config.auc_roc_path)
            if tracking_url_type_store != "file":
                        mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
            else:
                mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    obj = DataIngestion()
    raw_data_path,train_data_path,test_data_path = obj.initiate_data_ingestion()
    obj = DataValidation()
    validataion_status = obj.initiate_data_validation(raw_data_path)
    if(not validataion_status):
        print('Data validation is not successful.')
    else:
        obj = DataTransformation()
        train_array,test_array = obj.initiate_data_transformation(train_data_path,test_data_path)
        obj = ModelTraining()        
        model_path = obj.initiate_model_training(train_array,test_array)
        obj = ModelEvaluation()
        obj.initiate_model_evaluation(test_array,model_path)
        print('Model evaluation is successful.')
        
        
                    
                
                    
        

            