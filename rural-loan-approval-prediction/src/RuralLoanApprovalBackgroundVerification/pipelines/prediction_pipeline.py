import os
import sys
import pandas as pd
from src.RuralLoanApprovalBackgroundVerification.logger import logging
from src.RuralLoanApprovalBackgroundVerification.exception import customexception
from src.RuralLoanApprovalBackgroundVerification.utils.utils import *
from src.RuralLoanApprovalBackgroundVerification.components.data_validation import DataValidation
import mlflow
mlflow.autolog()

class PredictPipeline:
    def __init__(self)->None:
        pass
    def predict(self,features):
        try:
            # obj = DataValidation()
            # if obj.validate_all_columns(features):
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            # preprocessor_path = 'artifacts/preprocessor.pkl'
            # model_path = 'artifacts/model.pkl'
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            scaled_data = preprocessor.transform(features)
            pred = model.predict(scaled_data)
            return pred
            # else: 
            #     raise "Data is failed to be validated"
        except Exception as e:
            raise customexception(e,sys)
    
class CustomData:
    def __init__(self,
                 city: str,
                 age: int,
                 sex: str,
                 social_class: str,
                 primary_business: str,
                 secondary_business: str,
                 annual_income: float,
                 monthly_expenses: float,
                 old_dependents: int,
                 young_dependents: int,
                 home_ownership: float,
                 type_of_house: str,
                 occupants_count: int,
                 house_area: float,
                 sanitary_availability: float,
                 water_availabity: float,
                 loan_purpose: str,
                 loan_tenure: int,
                 loan_installments: int,
                 ):
        self.city = city
        self.age = age
        self.sex = sex
        self.social_class = social_class
        self.primary_business = primary_business
        self.secondary_business = secondary_business
        self.annual_income = annual_income
        self.monthly_expenses = monthly_expenses
        self.old_dependents = old_dependents
        self.young_dependents = young_dependents
        self.home_ownership = home_ownership
        self.type_of_house = type_of_house
        self.occupants_count = occupants_count
        self.house_area = house_area
        self.sanitary_availability = sanitary_availability
        self.water_availabity = water_availabity
        self.loan_purpose = loan_purpose
        self.loan_tenure = loan_tenure
        self.loan_installments = loan_installments
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                                            'city': [self.city],
                                            'age': [self.age],
                                            'sex': [self.sex],
                                            'social_class': [self.social_class],
                                            'primary_business': [self.primary_business],
                                            'secondary_business': [self.secondary_business],
                                            'annual_income': [self.annual_income],
                                            'monthly_expenses': [self.monthly_expenses],
                                            'old_dependents': [self.old_dependents],
                                            'young_dependents': [self.young_dependents],
                                            'home_ownership': [self.home_ownership],
                                            'type_of_house': [self.type_of_house],
                                            'occupants_count': [self.occupants_count],
                                            'house_area': [self.house_area],
                                            'sanitary_availability': [self.sanitary_availability],
                                            'water_availabity': [self.water_availabity],
                                            'loan_purpose': [self.loan_purpose],
                                            'loan_tenure': [self.loan_tenure],
                                            'loan_installments': [self.loan_installments],
                                            
                                        }

                df = pd.DataFrame(custom_data_input_dict)
                return df
            except Exception as e:
                logging.info('Exception occured in prediction pipeline.')
                raise customexception(e,sys)