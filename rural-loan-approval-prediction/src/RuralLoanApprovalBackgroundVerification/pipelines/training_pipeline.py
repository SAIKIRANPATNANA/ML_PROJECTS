import os
import sys
from src.RuralLoanApprovalBackgroundVerification.logger import logging
from src.RuralLoanApprovalBackgroundVerification.exception import customexception
import pandas as pd
from src.RuralLoanApprovalBackgroundVerification.components.data_ingestion import DataIngestion
from src.RuralLoanApprovalBackgroundVerification.components.data_transformation import DataTransformation
from src.RuralLoanApprovalBackgroundVerification.components.model_trainer import ModelTrainer
from src.RuralLoanApprovalBackgroundVerification.components.model_evaluation import ModelEvaluation
from src.RuralLoanApprovalBackgroundVerification.components.data_validation import DataValidation 
import mlflow
mlflow.autolog()

class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()
            return train_data_path,test_data_path
        except Exception as e:
            raise customexception(e,sys)
    def start_data_validation(self,df):
        try: 
            data_validation = DataValidation()
            validation_status = data_validation.validate_all_columns(df)
            return validation_status
        except:
            raise e
    def start_data_transformation(self,train_data_path,test_data_path):
        try:
            data_transformation = DataTransformation()
            train_array,test_array = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
            return train_array,test_array
        except Exception as e:
            raise customexception(e,sys)
    def start_model_training(self,train_array,test_array):
        try:
            model_trainer = ModelTrainer()
            model_trainer.initiate_model_training(train_array,test_array)
        except Exception as e:
            raise customexception(e,sys)
    def start_model_evaluation(self,train_array,test_array):
        try: 
            model_evaluation = ModelEvaluation()
            model_evaluation.initiate_model_evaluation(train_array,test_array)
        except Exception as e:
            raise customexception(e,sys)

def start_training():
    try:
        obj = TrainingPipeline()
        logging.info("Training pipeline started.")
        train_data_path,test_data_path = obj.start_data_ingestion()
        df = pd.read_csv("data/trainingData.csv")
        validation_status = obj.start_data_validation(df)
        train_array,test_array = obj.start_data_transformation(train_data_path,test_data_path)
        obj.start_model_training(train_array,test_array)
        obj.start_model_evaluation(train_array,test_array)
        logging.info("Training is successfully completed.")
    except Exception as e:
        raise customexception(e,sys)

if __name__ == "__main__":
    start_training()
    print("Training is successfully completed.")