import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
import sys,os
from src.utils import save_obj
from dataclasses import dataclass

@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformatoin Pipeline Initiated")
            num_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
            cat_cols = ['cut', 'color', 'clarity']
            cut_cats = [ 'Fair','Good', 'Very Good', 'Premium',  'Ideal']
            color_cats = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_cats = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']   
            num_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),('std_scaler',StandardScaler())])
            cat_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                        ('ordinalencoder',OrdinalEncoder(categories=[cut_cats,color_cats,clarity_cats])),('std_scaler',StandardScaler()) ])
            preprocessor=ColumnTransformer([('num_pipeline',num_pipeline,num_cols),('cat_pipeline',cat_pipeline,cat_cols)])
            logging.info('pipeline has been created')
            return preprocessor
        except Exception as e:
            logging.info("Exception Occured In Getting Object Of Data Transformation")
            raise CustomException(sys,e)
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info('Reading Train and Test Data has been completed')
            logging.info(f'Train DataFrame Head : \n {train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head : \n {test_df.head().to_string()}')
            logging.info('Obtaining Preprocessing Object')
            preprocessing_obj = self.get_data_transformation_object()
            target_col = 'price'
            drop_cols = [target_col,'id']
            input_features_train_df = train_df.drop(columns=drop_cols,axis=1)
            target_feature_train_df = train_df[target_col]
            input_features_test_df = test_df.drop(drop_cols,axis=1)
            target_feature_test_df = test_df[target_col]
            input_features_train_array = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_array = preprocessing_obj.transform(input_features_test_df)
            train_array = np.c_[input_features_train_array,np.array(target_feature_train_df)]
            test_array = np.c_[input_features_test_array,np.array(target_feature_test_df)]
            save_obj(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info('Data Transformation Has Been Completed.')
            return (train_array,test_array,self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            logging.info("Exception Occured In Initiating Data Transformation")
            raise CustomException(sys,e)
  
