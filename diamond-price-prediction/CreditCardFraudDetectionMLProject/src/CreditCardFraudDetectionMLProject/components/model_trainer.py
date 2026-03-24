import os 
import sys
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow 
from src.CreditCardFraudDetectionMLProject.logger import logging
from src.CreditCardFraudDetectionMLProject.exception import CustomException
from src.CreditCardFraudDetectionMLProject.components.data_ingestion import DataIngestion
from src.CreditCardFraudDetectionMLProject.components.data_validation import DataValidation
from src.CreditCardFraudDetectionMLProject.components.data_transformation import DataTransformation
from src.CreditCardFraudDetectionMLProject.utils.utils import *
from dataclasses import dataclass
import warnings as warn
warn.filterwarnings('ignore')
from xgboost import XGBClassifier
mlflow.autolog()

@dataclass 
class ModelTrainingConfig:
    model_path = os.path.join('artifacts','model.pkl')
class ModelTraining:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()
    def initiate_model_training(self,train_array,test_array):
        try: 
            x_train = train_array[:,:-1]
            y_train = train_array[:,-1]
            x_test = test_array[:,:-1]
            y_test = test_array[:,-1]
            model = XGBClassifier()
            model.fit(x_train,y_train)
            save_object(
                obj = model,
                file_path = self.model_training_config.model_path
            )
            return self.model_training_config.model_path
        except CustomException as  e:
            raise (e,sys)

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
        print('Model training is successful.')

            
            
                        
                    
                    