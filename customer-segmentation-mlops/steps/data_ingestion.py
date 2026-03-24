import pandas as pd
import logging
from zenml import step

class DataIngestion:
    def __init__(self)->None:
        pass
    def get_data(self)->pd.DataFrame:
        df = pd.read_csv('dataset/olist_customers_dataset.csv')
        return df

@step 
def ingest_data()->pd.DataFrame:
    try: 
        data_ingestion = DataIngestion()
        df = data_ingestion.get_data()
        return df 
    except Exception as e:
        logging.error(e)
        raise e
    
# if __name__ == '__main__':
#     df = ingest_data()
