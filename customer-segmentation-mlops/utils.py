import logging 
import pandas as pd 
from model.data_cleaner import *

def get_test_data():
    try:
        data = pd.read_csv('/home/user/Documents/ML_DL_PROJECTS/MLOPSProject/dataset/olist_customers_dataset.csv')
        data = data.sample(n=100)
        data_preprocessing = DataPreprocessing()
        data_cleaner = DataCleaning(data,data_preprocessing)
        preprocessed_data = data_cleaner.handle_data()
        data = preprocessed_data.drop(['review_score'],axis=1)
        data = data.to_json(orient='split')
        return data
    except Exception as e:
        logging.error(e)
        raise e 

