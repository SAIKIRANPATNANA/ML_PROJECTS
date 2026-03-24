import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
if __name__ == '__main__':
    data_ingestion = DataIngestion()
    data_transformation = DataTransformation()
    train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()
    train_array,test_array,preprocessing_obj_file_path=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_array,test_array)
    # print(train_data_path,test_data_path)
    # print(preprocessing_obj_file_path)

