import os,sys
import pandas as pd
import numpy as np
from src.logger import logging 
from src.exception import CustomException
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from src.utils import save_obj
from src.utils import evaluate_model
from dataclasses import dataclass

@dataclass 
class ModelTrainerCongig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerCongig()
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting the dependent and the independent data from the train and test datasets")
            x_train,y_train,x_test,y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])
            models = {'linear_regressor': LinearRegression(), 
                      'lasso_regressor': Lasso(), 
                      'Ridge_regressor': Ridge(),
                    #   'Decision_tree_regressor':DecisionTreeRegressor(),
                    #   'Support_vector_regressor':SVR(),
                    #   'K_neighbors_regressor':KNeighborsRegressor(),
                       'Random_forest_regressor':RandomForestRegressor() }
            model_report:dict = evaluate_model(x_train,y_train,x_test,y_test,models)
            print(model_report)
            logging.info(f'Model Report : {model_report}')
            best_model_score = max(sorted(list(model_report.values())))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            print('\n====================================================================================\n')
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            save_obj(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
        except Exception as e:
            logging.info("Exception Has Been Occured In Initiating The Model Training")
            raise CustomException(e,sys)
        

