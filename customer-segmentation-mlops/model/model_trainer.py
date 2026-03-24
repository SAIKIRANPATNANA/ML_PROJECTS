import logging
from abc import ABC, abstractmethod
import optuna 
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression 

class ModelTraining(ABC):
    @abstractmethod
    def train(self,x_train,y_train):
        pass 
    @abstractmethod
    def optimize(self,trial,x_train,y_train,x_test,y_test):
        pass

class RandomForestRegressorModel(ModelTraining):
    def train(self,x_train,y_train,**kwargs):
        model = RandomForestRegressor(**kwargs)
        model.fit(x_train,y_train)
        return model
    def optimize(self,trial,x_train,y_train,x_test,y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        model = self.train(x_train, y_train, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
        return model.score(x_test, y_test)

class LinearRegressorModel(ModelTraining):
    def train(self,x_train,y_train,**kwargs):
        model = LinearRegression(**kwargs)
        model.fit(x_train,y_train)
        return model
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        model = self.train(x_train, y_train)
        return model.score(x_test, y_test)

class HyperParameterTuning:
    def __init__(self,model,x_train,y_train,x_test,y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test 
        self.y_test = y_test
    def optimize(self,n_trials=1):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.model.optimize(trial, self.x_train, self.y_train, self.x_test, self.y_test), n_trials=n_trials)
        return study.best_trial.params


    
     
