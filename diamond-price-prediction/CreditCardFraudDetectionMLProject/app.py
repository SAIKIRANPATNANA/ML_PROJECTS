import streamlit as st
import pandas as pd 
import numpy as  np 
import warnings as warn 
warn.filterwarnings("ignore")
from src.CreditCardFraudDetectionMLProject.pipelines.prediction_pipeline import *

st.markdown("<h1>Credit Card Fraud Detection💳🕵🏻</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; font-size:30px; text-align:center;'>Developed By <span style='color:Red ;font-size:35px;'>Sai Kiran Patnana</span></h3>",unsafe_allow_html=True)
LIMIT_BAL = st.number_input('LIMIT_BALANCE',min_value=10000,max_value=700000,format='%d')
SEX = st.selectbox('SEX',['Male','Female'])
if(SEX=='Male'):
    SEX = 1
else:
    SEX = 2
EDUCATION = st.selectbox('EDUCATION LEVEL',[1,2,3,4,5,6])
MARRIAGE = st.selectbox('MARRIAGES',[0,1,2,3])
AGE = st.slider('AGE',21,75,21)
PAY_0 = st.selectbox('PAY_0',[-1,  0, -2,  1,  2,  3,  4,  8])
PAY_2 = st.selectbox('PAY_2',[ 0, -1, -2,  2,  3,  5,  7,  4,  1])
PAY_3 = st.selectbox('PAY_3',[-1,  0,  2, -2,  3,  4,  6,  7,  1,  5])
PAY_4 = st.selectbox('PAY_4',[ 0, -2, -1,  2,  3,  4,  5,  7])
PAY_5 = st.selectbox('PAY_5',[ 0, -1,  2, -2,  3,  5,  4,  7])
PAY_6 = st.selectbox('PAY_6', [0, -1,  2, -2,  3,  6,  4,  7])
BILL_AMT1 = st.number_input('BILL_AMT1',min_value=-14386,max_value=507726)
BILL_AMT2 = st.number_input('BILL_AMT2',min_value=-13543,max_value=509229)
BILL_AMT3 = st.number_input('BILL_AMT3',min_value=-9850,max_value=499936)
BILL_AMT4 = st.number_input('BILL_AMT4',min_value=-3684,max_value=628699)
BILL_AMT5 = st.number_input('BILL_AMT5',min_value=-28335,max_value=484612)
BILL_AMT6 = st.number_input('BILL_AMT6',min_value=-339603,max_value=473944)
PAY_AMT1 = st.number_input('PAY_AMT1',min_value=1000,max_value=199646)
PAY_AMT2 = st.number_input('PAY_AMT2',min_value=390,max_value=285138)
PAY_AMT3 = st.number_input('PAY_AMT3',min_value=228,max_value=133657)
PAY_AMT4 = st.number_input('PAY_AMT4',min_value=148,max_value=188840)
PAY_AMT5 = st.number_input('PAY_AMT5',min_value=1306,max_value=195599)
PAY_AMT6 = st.number_input('PAY_AMT6',min_value=0, max_value=528666)
data = CustomData(LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2,
       PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2,
       BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1,
       PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6)
final_data = data.get_data_as_dataframe()
prediction_pipeline = PredictionPipeline()
pred = prediction_pipeline.predict(final_data) 
if st.button("Predict Fraud"):
        if int(pred)>0.5:
            st.warning("It's a Fraud..!")
        else:
            st.info("It's not a Fruud")
               