import logging
import mlflow
import pandas as pd
import numpy as np 
from model.model_trainer import HyperParameterTuning,RandomForestRegressorModel,LinearRegressorModel
from sklearn.base import RegressorMixin 
from zenml import step 
# from data_ingestion import ingest_data
# from data_transformation import transform_data
from zenml.client import Client
from config import ModelConfig 

experiment_tracker = Client().active_stack.experiment_tracker
config = ModelConfig()
class ModelTraining:
    def __init__(self)->None:
        pass
    def get_trained_model(self,x_train:pd.DataFrame,y_train:pd.Series,x_test:pd.DataFrame,y_test:pd.Series)->RegressorMixin:
        try:
            model = None
            tuner = None
            if(config.model_name=='RandomForestRegressor'):
                mlflow.sklearn.autolog()
                model = RandomForestRegressorModel()
            elif(config.model_name=='LinearRegressor'):
                mlflow.sklearn.autolog()
                model = LinearRegressorModel()
            else:
                raise ValueError('Sorry, your model name is not supported')
            if(config.fine_tuning):
                tuner = HyperParameterTuning(model,x_train,y_train,x_test,y_test)
                best_params = tuner.optimize()
                trained_model = model.train(x_train,y_train,**best_params)
            else:
                trained_model = model.train(x_train,y_train)
            return trained_model
        except Exception as e:
            logging.error(e)
            raise e

@step(experiment_tracker=experiment_tracker.name)
def train_model(x_train:pd.DataFrame,y_train:pd.Series,x_test:pd.DataFrame,y_test:pd.Series)->RegressorMixin:
    try:
        model_training = ModelTraining()
        trained_model = model_training.get_trained_model(x_train,y_train,x_test,y_test)
        return trained_model
    except Exception as e:
        logging.error(e)
        raise e

# if __name__ == '__main__':
#     df = ingest_data()
#     x_train,x_test,y_train,y_test = transform_data(df)
#     trained_model = train_model(x_train,y_train,x_test,y_test)

