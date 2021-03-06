import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle

st.title("Predict heart disease")
st.text("Enter x covariates")

model = pickle.load(open("LogisticRegression", "rb"))

# st.write(str(model))

age = st.text_input("Age")
sex = st.selectbox("Sex", ["0", "1"])
cp = st.selectbox("Chest Pain", ["0", "1"])
trestbps = st.text_input(label="Resting Blood Pressure")
chol = st.text_input(label="Serum Cholestoral in mg/dl")
fbs = st.selectbox("is fasting blood Sugar > 120 mg/dl", ["0", "1"])
restecg = st.selectbox(
    'Resting Electrocardiographic Results :',
    ["0", "1", "2"])
thalach = st.text_input(label="Maximum Heart Rate Achieved")
exang = st.selectbox(
    'exercise induced angina :',
    ["0", "1"])
oldpeak = st.text_input(label="ST depression induced by exercise relative to rest")
slope = st.selectbox(
    'The slope of the peak exercise ST segment :',
    ["0", "1", "2"])
ca = st.selectbox("number of major vessels (0-3) colored by fluorosopy", ["0", "1", "2", "3"])

#: 3 = normal; 6 = fixed defect; 7 = reversable defect
thal = st.selectbox("Thal", ["1", "2", "3", "6", "7"])

# Prediction inputs to model.predict are array of arrays.  one array here.
pred = model.predict([[#int(age),
                       float(cp),
                       int(trestbps),
                       int(chol),
                       int(fbs),
                       int(restecg),
                       int(thalach),
                       int(exang),
                       float(oldpeak),
                       int(slope),
                       int(ca),
                       int(thal)]])

if st.button("Check the Patient"):
    with st.spinner("Predicting result... "):
        time.sleep(2)
    if pred == 0:
        st.header("Patient has a heart problem")
    else:
        st.header("Patient is healthy")




