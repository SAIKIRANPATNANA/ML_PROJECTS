import logging
from abc import ABC, abstractmethod
from typing import Union 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataCleaningMethod(ABC) :
    @abstractmethod
    def handle_data(self,data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass 

class DataPreprocessing(DataCleaningMethod):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            df = data.copy()
            cols_to_drop =  ["order_approved_at",
                            "order_delivered_carrier_date",
                            "order_delivered_customer_date",
                            "order_estimated_delivery_date",
                            "order_purchase_timestamp",
                            "customer_zip_code_prefix", 
                            "order_item_id" ]
            df.drop(cols_to_drop,axis=1,inplace=True)
            df["product_weight_g"].fillna(df["product_weight_g"].median(), inplace=True)
            df["product_length_cm"].fillna(df["product_length_cm"].median(), inplace=True)
            df["product_height_cm"].fillna(df["product_height_cm"].median(), inplace=True)
            df["product_width_cm"].fillna(df["product_width_cm"].median(), inplace=True)
            df["review_comment_message"].fillna("No Review",inplace=True)
            df = df.select_dtypes(include=[np.number])
            return df  
        except Exception as e:
            logging.error(e)
            raise e
class DataSplitting(DataCleaningMethod):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:          
        try:
            df = data.copy()
            x = df.drop("review_score", axis=1)
            y = df['review_score']
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42) 
            return x_train,x_test,y_train,y_test
        except Exception as e:
            logging.error(e)
            raise e

class DataCleaning:
    def __init__(self, data: pd.DataFrame, method: DataCleaningMethod):
        self.data = data 
        self.method = method 
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:  
        try:    
            return self.method.handle_data(self.data)
        except Exception as e:
            raise e


