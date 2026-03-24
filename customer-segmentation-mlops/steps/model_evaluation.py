import logging
import mlflow 
import zenml
import pandas as pd 
import numpy as np 
from model.model_evaluator import  RootMeanSquaredError,MeanSquaredError,MeanAbsoluteError,R2Score
from sklearn.base import RegressorMixin 
from typing_extensions import Annotated 
from zenml import step 
# from data_ingestion import ingest_data
# from data_transformation import transform_data
# from model_training import train_model
from zenml.client import Client
from typing import Tuple
from config import ModelConfig
experiment_tracker = Client().active_stack.experiment_tracker

class ModelEvaluation:
    def __init__(self):
        pass
    def get_evaluation_scores(self,model:RegressorMixin,x_test:pd.DataFrame,y_test: pd.Series)->Tuple[Annotated[float,'r2_score'],Annotated[float,'mse'],Annotated[float,'rmse'], Annotated[float,'mae']]:
        try:
            y_pred = model.predict(x_test)
            mean_squared_error = MeanSquaredError()
            mse = mean_squared_error.calculate_score(y_test,y_pred)
            mlflow.log_metric("mse", mse)
            root_mean_squared_error = RootMeanSquaredError()
            rmse = root_mean_squared_error.calculate_score(y_test,y_pred)
            mlflow.log_metric("rmse", rmse)
            mean_absolute_error = MeanAbsoluteError()
            mae = mean_absolute_error.calculate_score(y_test,y_pred)
            mlflow.log_metric("mae", mae)
            r2 = R2Score()
            r2_score = r2.calculate_score(y_test,y_pred)
            mlflow.log_metric("r2_score", r2_score)
            return mse, rmse, mae, r2_score
        except Exception as e: 
            logging.error(e)
            raise e

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model:RegressorMixin, x_test:pd.DataFrame,y_test: pd.Series)->Tuple[Annotated[float,'r2_score'],Annotated[float,'mse'],Annotated[float,'rmse'], Annotated[float,'mae']]:
    try:
        model_evaluation = ModelEvaluation()
        mse,rmse,mae,r2_score = model_evaluation.get_evaluation_scores(model,x_test,y_test)
        print(f"mse: {mse}\nrmse: {rmse}\nmae: {mae}\nr2_score: {r2_score}\n")
        return mse,rmse,mae,r2_score
    except Exception as e:
        logging.error(e)
        raise e


# if __name__ == '__main__':

#     print(1)
#     df = ingest_data()
#     print(2)
#     x_train,x_test,y_train,y_test = transform_data(df)
#     print(3)
#     trained_model = train_model(x_train,y_train,x_test,y_test)
#     print(4)
#     mse,rmse,mae,r2_score = evaluate_model(trained_model,x_test,y_test)
