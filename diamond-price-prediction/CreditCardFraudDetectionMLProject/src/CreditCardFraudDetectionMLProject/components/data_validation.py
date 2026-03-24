import os 
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from src.CreditCardFraudDetectionMLProject.logger import logging
from src.CreditCardFraudDetectionMLProject.exception import CustomException
from src.CreditCardFraudDetectionMLProject.components.data_ingestion import DataIngestion
from src.CreditCardFraudDetectionMLProject.utils.utils import *
from dataclasses import dataclass
import warnings as warn
warn.filterwarnings('ignore')
import mlflow 
mlflow.autolog()

class DataValidation:
    def __init__(self):
        pass
    def  initiate_data_validation(self,raw_data_path):
        try: 
            req_cols = ['LIMIT_BAL',
                        'SEX',
                        'EDUCATION',
                        'MARRIAGE',
                        'AGE',
                        'PAY_0',
                        'PAY_2',
                        'PAY_3',
                        'PAY_4',
                        'PAY_5',
                        'PAY_6',
                        'BILL_AMT1',
                        'BILL_AMT2',
                        'BILL_AMT3',
                        'BILL_AMT4',
                        'BILL_AMT5',
                        'BILL_AMT6',
                        'PAY_AMT1',
                        'PAY_AMT2',
                        'PAY_AMT3',
                        'PAY_AMT4',
                        'PAY_AMT5',
                        'PAY_AMT6',
                        'default payment next month']
            df = pd.read_csv(raw_data_path)
            act_cols = df.columns
            if(len(req_cols)!=len(act_cols)):
                return False
            if(set(req_cols) != set(act_cols)):
                return False
            return True 
        except CustomException as e:
            raise(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    raw_data_path,_,_ = obj.initiate_data_ingestion()
    obj = DataValidation()
    validataion_status = obj.initiate_data_validation(raw_data_path)
    if(validataion_status):
        print('Data validation is successful.')
    
    

