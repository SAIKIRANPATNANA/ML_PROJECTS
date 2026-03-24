import os
import sys
import pandas as pd
import numpy as np
import warnings as warn 
warn.filterwarnings('ignore')
from src.RuralLoanApprovalBackgroundVerification.logger import logging
from src.RuralLoanApprovalBackgroundVerification.exception import customexception
from dataclasses import dataclass
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split
from src.RuralLoanApprovalBackgroundVerification.components.data_ingestion import DataIngestion
from src.RuralLoanApprovalBackgroundVerification.utils.utils import *
from typing_extensions import Annotated 
from typing import Tuple
import mlflow
mlflow.autolog()

@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts','preprocessor.pkl')
class DataTransformation:
    def __init__(self)->None:
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformation(self)->None:
        try:
            logging.info('Data transformation started.')
            categorical_cols = ['sex','type_of_house','loan_purpose',]
            numerical_cols = ['age', 'annual_income', 'monthly_expenses', 'old_dependents','young_dependents', 'home_ownership', 'occupants_count', 'house_area',
                            'sanitary_availability', 'water_availabity', 'loan_tenure','loan_installments']
            sex_categories = ['TG','M','F']
            type_of_house_categories = ['R', 'T1', 'T2']
            loan_purpose_categories = ['Apparels',
                                        'Agro Based Businesses',
                                        'Animal husbandry',
                                        'Meat Businesses',
                                        'Handicrafts',
                                        'Farming/ Agriculture',
                                        'Education Loan',
                                        'Retail Store',
                                        'Eateries',
                                        'Business Services - II',
                                        'Tobacco Related Activities',
                                        'Construction Related Activities',
                                        'Retail Sale',
                                        'Artifical Jewellry Selling',
                                        'Carpentery work',
                                        'Food Items',
                                        'Business Services - I',
                                        'Transportation Services',
                                        'Flower Business',
                                        'Beauty Salon',
                                        'Repair Services',
                                        'Laundry Services',
                                        'Agarbatti Business',
                                        'Utensil Selling',
                                        'Sanitation',
                                        'Recycling/ Waste Management',
                                        'Others',
                                        'Vocational Loans',
                                        'Jewellry Shop',
                                        'Training',
                                        'Miscellaneous',
                                        'Cyber Caf_',
                                        'Tent Services',
                                        'Cable TV Services',
                                        'Professional',
                                        'Tuition Centre',
                                        'Manufacturing']
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='mean')),
                ('scaler',StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[sex_categories,type_of_house_categories,loan_purpose_categories])),
                ('scaler',StandardScaler())
                ]
            )
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            return preprocessor
        except Exception as e:
            logging.info("Exception occured in data transformation.")
            raise customexception(e,sys)
    def initiate_data_transformation(self,train_path:str,test_path:str)->Tuple[
                            Annotated[np.array, "train_array"],
                            Annotated[np.array, "test_array"]]:
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            df = pd.concat([train_df,test_df],ignore_index=True)
            drop_columns = ['city','primary_business','secondary_business','social_class']
            df = df.drop(drop_columns,axis=1)
            num_df = df.select_dtypes(exclude='object')
            Q1 = num_df.quantile(.10)
            Q3 = num_df.quantile(.90)
            IQR = Q3 - Q1
            outliers_iqr = ((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)
            outliers_indices_iqr = outliers_iqr[outliers_iqr].index
            df = df.drop(outliers_indices_iqr)
            df = df.reset_index(drop=True)
            train_df,test_df = train_test_split(df,test_size=0.25,random_state=42)
            target_column_name = 'loan_amount'
            preprocessor = self.get_data_transformation()
            input_train_df = train_df.drop(target_column_name,axis=1)
            target_train_df = train_df[target_column_name]
            input_test_df = test_df.drop(target_column_name,axis=1)
            target_test_df = test_df[target_column_name]
            input_train_array = preprocessor.fit_transform(input_train_df)
            input_test_array = preprocessor.transform(input_test_df)
            save_object(
                file_path = self.data_transformation_config.preprocessor_file_path,
                obj = preprocessor
            )
            train_array = np.c_[input_train_array, np.array(target_train_df)]
            test_array = np.c_[input_test_array, np.array(target_test_df)]
            logging.info("Data transromation is successfully compeleted.")
            return (
                train_array,
                test_array
            )
        except Exception as e:
            logging.info("Exception occured in data transromation.")
            raise customexception(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    obj = DataTransformation()
    train_array,test_array = obj.initiate_data_transformation(train_data_path,test_data_path)
    print("Data transromation is successfully compeleted.")



 