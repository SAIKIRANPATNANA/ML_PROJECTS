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
from src.CreditCardFraudDetectionMLProject.components.model_trainer import ModelTraining
from src.CreditCardFraudDetectionMLProject.components.model_evaluation import ModelEvaluation
from src.CreditCardFraudDetectionMLProject.utils.utils import *
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow 
mlflow.autolog()
import warnings as warn
warn.filterwarnings('ignore')

class TrainingPipeline:
    def __init__(self):
        pass 
    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            raw_data_path,train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()
            return raw_data_path,train_data_path,test_data_path
        except Exception as e:
            raise CustomException(e,sys)
    def start_data_validation(self,raw_data_path):
        try: 
            data_validation = DataValidation()
            validataion_status = data_validation.initiate_data_validation(raw_data_path)
            return validataion_status 
        except Exception as e:
            raise CustomException(e,sys)
    def start_data_transformation(self,train_data_path,test_data_path):
        try: 
            data_transformation = DataTransformation()
            train_array,test_array = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
            return train_array,test_array
        except Exception as e:
            raise (e,sys)
    def start_model_training(self,train_data_path,test_data_path):
        try: 
            model_trainer = ModelTraining()
            model_path = model_trainer.initiate_model_training(train_array,test_array)
            return model_path
        except Exception as e:
            raise CustomException(e,sys)
    def start_model_evaluation(self,test_array,model_path):
        try:
            model_evaluation = ModelEvaluation()
            model_evaluation.initiate_model_evaluation(test_array,model_path)
        except Exception as e: 
            raise CustomException(e,sys)

if __name__ == '__main__':
    trainer = TrainingPipeline()
    raw_data_path,train_data_path,test_data_path = trainer.start_data_ingestion()
    validataion_status = trainer.start_data_validation(raw_data_path)
    if(validataion_status):
        train_array,test_array = trainer.start_data_transformation(train_data_path,test_data_path)
        model_path = trainer.start_model_training(train_array,test_array)
        trainer.start_model_evaluation(test_array,model_path)
        print('Training is successful.')
    else:
        print('Data validation is not successful.')