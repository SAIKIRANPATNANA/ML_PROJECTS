import logging
from typing import Tuple
import pandas as pd
import numpy as np
from model.data_cleaner import DataCleaning,DataPreprocessing,DataSplitting
from typing_extensions import Annotated 
# from data_ingestion import ingest_data
from zenml import step 


class DataTransformation:
    def __init__(self)->None:
        pass
    def get_preprocessed_data(self,data: pd.DataFrame) ->  Tuple[
                            Annotated[pd.DataFrame, "x_train"],
                            Annotated[pd.DataFrame, "x_test"],
                            Annotated[pd.Series, "y_train"],
                            Annotated[pd.Series, "y_test"] ]:
                data_preprocessing = DataPreprocessing()
                data_cleaner = DataCleaning(data,data_preprocessing)
                preprocessed_data = data_cleaner.handle_data()
                data_splitting = DataSplitting()
                data_cleaner = DataCleaning(preprocessed_data,data_splitting)
                x_train,x_test,y_train,y_test = data_cleaner.handle_data()
                return x_train,x_test,y_train,y_test


@step 
def transform_data(data: pd.DataFrame) -> Tuple[
                            Annotated[pd.DataFrame, "x_train"],
                            Annotated[pd.DataFrame, "x_test"],
                            Annotated[pd.Series, "y_train"],
                            Annotated[pd.Series, "y_test"] ]:
            try: 
                data_transformation = DataTransformation() 
                x_train,x_test,y_train,y_test = data_transformation.get_preprocessed_data(data)
                return x_train,x_test,y_train,y_test
            except Exception as e:
                logging.error(e)
                raise e
                
# if __name__ == '__main__':
#     df = ingest_data()
#     x_train,y_train,x_test,y_test = transform_data(df)
