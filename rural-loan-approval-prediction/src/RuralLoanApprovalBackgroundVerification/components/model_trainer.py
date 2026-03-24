import pandas as pd
import numpy as np
import os
import sys
import warnings as warn 
warn.filterwarnings('ignore')
from src.RuralLoanApprovalBackgroundVerification.logger import logging
from src.RuralLoanApprovalBackgroundVerification.exception import customexception
from dataclasses import dataclass
from src.RuralLoanApprovalBackgroundVerification.utils.utils import *
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from src.RuralLoanApprovalBackgroundVerification.components.data_ingestion import DataIngestion
from src.RuralLoanApprovalBackgroundVerification.components.data_transformation import DataTransformation
import mlflow 
mlflow.autolog()
@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
class ModelTrainer:
    def __init__(self)->None:
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Model training started.")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet(),
            'Randomforest':RandomForestRegressor(),
            'Xgboost':XGBRegressor(),
            }
            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            print('\n====================================================================================\n')
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}.')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    obj = DataTransformation()
    train_array,test_array = obj.initiate_data_transformation(train_data_path,test_data_path)
    obj = ModelTrainer()
    obj.initiate_model_training(train_array,test_array)
    print("Model training is successfully compeleted.")


        
    