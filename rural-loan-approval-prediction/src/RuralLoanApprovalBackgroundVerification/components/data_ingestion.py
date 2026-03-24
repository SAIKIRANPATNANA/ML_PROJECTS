import os
import sys
import pandas as pd
import numpy as np
import warnings as warn 
warn.filterwarnings('ignore')
from src.RuralLoanApprovalBackgroundVerification.logger import logging
from src.RuralLoanApprovalBackgroundVerification.exception import customexception
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
from typing_extensions import Annotated 
from typing import Tuple
import mlflow
mlflow.autolog()

@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self)->None:
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self)->Tuple[
                            Annotated[str, "train_data_path"],
                            Annotated[str, "test_data_path"],
                            ]:
        logging.info("Data ingestion started.")
        try:
            # data = pd.read_csv('/home/user/Documents/ML_DL_PROJECTS/RuralLoanApprovalBackgroundVerificationProject/data/trainingData.csv')
            data = pd.read_csv("/RuralloanApprovalBackgroundVerificationProject/data/trainingData.csv")
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            train_data,test_data=train_test_split(data,test_size=0.25,random_state=42)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info("Data ingestion completed.")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
           logging.info("Exception occured at data ingestion stage.")
           raise customexception(e,sys)

    
if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    print('Data Ingestion is successfully completed.')
        