import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

features = ['age', 'systolic_bp', 'diastolic_bp', 'cholesterol']
BEST_MODEL = 'Logistic Regression'

st.set_page_config(page_title='Diabetic Retinopathy Predictor', page_icon='🩺', layout='centered')
st.title('Diabetic Retinopathy Risk Predictor')
st.markdown('P653 Project | Enter patient values below to predict retinopathy risk.')
st.divider()

st.sidebar.header('About')
st.sidebar.markdown(f'Model: {BEST_MODEL} | Dataset: 6,000 patients | Features: Age, BP, Cholesterol')

st.subheader('Patient Information')
col1, col2 = st.columns(2)
with col1:
    age          = st.slider('Age (years)', 30, 110, 60)
    systolic_bp  = st.slider('Systolic BP (mmHg) - Normal: below 120', 60, 180, 110)
with col2:
    diastolic_bp = st.slider('Diastolic BP (mmHg) - Normal: below 80', 50, 140, 85)
    cholesterol  = st.slider('Cholesterol (mg/dl) - Normal: 125-200', 60, 160, 120)

st.divider()
if st.button('Predict Retinopathy Risk', use_container_width=True, type='primary'):
    patient = pd.DataFrame([[age, systolic_bp, diastolic_bp, cholesterol]], columns=features)
    if BEST_MODEL == 'Logistic Regression':
        patient_input = scaler.transform(patient)
    else:
        patient_input = patient
    prob  = model.predict_proba(patient_input)[0][1]
    label = 'Retinopathy' if prob >= 0.5 else 'No Retinopathy'
    st.subheader('Prediction Result')
    if prob >= 0.70:
        st.error(f'HIGH RISK - {label}  ({prob*100:.1f}% probability)')
        st.warning('Recommend immediate ophthalmologist referral.')
    elif prob >= 0.50:
        st.warning(f'MEDIUM RISK - {label}  ({prob*100:.1f}% probability)')
        st.info('Recommend follow-up within 3-6 months.')
    else:
        st.success(f'LOW RISK - {label}  ({prob*100:.1f}% probability)')
        st.info('Continue routine annual check-up.')
    fig, ax = plt.subplots(figsize=(5, 1.5))
    ax.barh(['Risk'], [prob], color='#F44336' if prob >= 0.5 else '#2196F3', height=0.4)
    ax.barh(['Risk'], [1-prob], left=[prob], color='#e0e0e0', height=0.4)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Probability')
    ax.set_title(f'Retinopathy Probability: {prob*100:.1f}%')
    st.pyplot(fig)
