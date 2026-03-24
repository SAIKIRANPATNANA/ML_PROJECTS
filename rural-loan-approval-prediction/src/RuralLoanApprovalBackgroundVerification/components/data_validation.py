import os
import sys
import pandas as pd
from src.RuralLoanApprovalBackgroundVerification.logger import logging
from src.RuralLoanApprovalBackgroundVerification.exception import customexception

class DataValidation:
    def __init__(self):
        pass

    def validate_all_columns(self, check_df: pd.DataFrame) -> bool:
        try:
            logging.info("Data validation Started.")
            refer_df = pd.read_csv("/RuralLoanApprovalBackgroundVerificationProject/data/trainingData.csv")
            # refer_df = pd.read_csv('/home/user/Documents/ML_DL_PROJECTS/RuralLoanApprovalBackgroundVerificationProject/data/trainingData.csv')
            req_cols = list(refer_df.columns)
            all_cols = list(check_df.columns)
            if len(req_cols) != len(all_cols):
                return False
            if set(req_cols) != set(all_cols):
                return False
            # if not check_df.dtypes.equals(refer_df[req_cols].dtypes):
            #     return False
            logging.info("Data validation is successfully completed.")
            return True
        except Exception as e:
            raise customexception(e, sys)

if __name__ == "__main__":
    df = pd.read_csv("/RuralLoanApprovalBackgroundVerificationProject/data/trainingData.csv")
    # df = pd.read_csv('/home/user/Documents/ML_DL_PROJECTS/RuralLoanApprovalBackgroundVerificationProject/data/trainingData.csv')
    obj = DataValidation()
    print(obj.validate_all_columns(df))
