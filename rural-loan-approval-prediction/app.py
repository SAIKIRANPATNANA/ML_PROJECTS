import streamlit as st
import pandas as pd 
import numpy as  np 
import warnings as warn 
warn.filterwarnings("ignore")
from src.RuralLoanApprovalBackgroundVerification.pipelines.prediction_pipeline import *

df = pd.read_csv('data/trainingData.csv')
social_class_options = list(df['social_class'].unique())
social_class_options.remove(np.nan)
primary_business_options = list(df['primary_business'].unique())
primary_business_options.remove(np.nan)
secondary_business_options = list(df['secondary_business'].unique())
secondary_business_options.remove(np.nan)
type_of_house_options = list(df['type_of_house'].unique())
type_of_house_options.remove(np.nan)
loan_purpose_options = list(df['loan_purpose'].unique())
loan_purpose_options.remove(np.nan)

st.title("  Loan Estimator for Rural People...")
st.markdown("<h3 style='text-align:center; font-size:20px; text-align:center;'>Developed By <span style='color:skyblue ;font-size:25px;'>Sai Kiran Patnana</span></h3>",unsafe_allow_html=True)

city = st.text_input("Village:",)
age = st.slider("Age:", 18, 100, 25)
sex = st.selectbox("Sex:", ["Male", "Female","Transgender"])
if sex=="Male":
    sex = 'M'
elif sex=='Female':
    sex = 'F'
else:
    sex = 'TG'
social_class = st.selectbox("Social Class:",social_class_options)
primary_business = st.selectbox("Primary Business:", primary_business_options)
secondary_business = st.selectbox("Secondary Business:", secondary_business_options) 
annual_income = st.number_input("Annual Income:", min_value=0.0, format="%f")
monthly_expenses = st.number_input("Monthly Expenses:", min_value=0.0, format="%f")
old_dependents = st.slider("Old Dependents:", 0, 10, 0)
young_dependents = st.slider("Young Dependents:", 0, 10, 0)
home_ownership = st.selectbox("Home Ownership:", ["Own", "Rent"])
if(home_ownership == 'Own'):
    home_ownership = 1.0
else:
    home_ownership = 0.0
type_of_house = st.selectbox("Type of House:",type_of_house_options)
occupants_count = st.slider("Occupants Count:", 1, 10, 1)
house_area = st.number_input("House Area:", min_value=0.0, format="%f")
sanitary_availability = st.selectbox("Sanitary Availability:", ["Yes", "No"])
if sanitary_availability == 'Yes':
    sanitary_availability = 1.0
else:
    sanitary_availability = 0.0
water_availabity = st.selectbox("Water Availability:", ["Yes", "No"])
if water_availabity == 'Yes':
    water_availabity = 1.0
else: 
    water_availabity = 0.0
loan_purpose = st.selectbox("Loan Purpose:", loan_purpose_options)
loan_tenure = st.slider("Loan Tenure (months):", 1, 120, 12)
loan_installments = st.slider("Loan Installments:", 1, 60, 12)

data = CustomData(city,
                 age,
                 sex,
                 social_class,
                 primary_business,
                 secondary_business,
                 annual_income,
                 monthly_expenses,
                 old_dependents,
                 young_dependents,
                 home_ownership,
                 type_of_house,
                 occupants_count,
                 house_area,
                 sanitary_availability,
                 water_availabity,
                 loan_purpose,
                 loan_tenure,
                 loan_installments)

final_data = data.get_data_as_dataframe()
predict_pipeline = PredictPipeline()
pred=predict_pipeline.predict(final_data)

if st.button("Estimate"):
        st.success(f"Estimated Loan Amount is  {round(pred[0],2)}")

