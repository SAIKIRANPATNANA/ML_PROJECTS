import streamlit as st
import os
from RedWineQualityMLProject.pipeline.prediction_pipeline import PredictionPineline
st.title('Wine Quality Prediction ')
fixed_acidity = st.number_input('Fixe Acidity')
volatile_acidity =  st.number_input('Volatile Acidiy')
citric_acid = st.number_input('Citric Acid')
residual_sugar =  st.number_input('Residual Sugar')
chlorides = st.number_input('Chlorides')
free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide')
total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide')
density = st.number_input('Density')
pH = st.number_input('pH')
sulphates = st.number_input('Sulphates')
alcohol = st.number_input('Alcohol')
input_data = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]]
if st.button('Train Model'):
    os.system('python main.py')
    st.write('Model Trained Successfully')
if st.button('Predict Quality'):
    prediction_pipeline = PredictionPineline()
    st.write(f'Esmated Quality: {prediction_pipeline.predict(input_data)}')
