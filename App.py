import streamlit as st
import pandas as pd
from main import load_model, predict_disease

st.title("AI Disease Predictor")

st.write("Enter patient information:")

age = st.slider("Age", 0, 100, 30)
fever = st.radio("Fever", [0, 1])
cough = st.radio("Cough", [0, 1])
headache = st.radio("Headache", [0, 1])

if st.button("Predict"):
    model = load_model()
    result = predict_disease(age, fever, cough, headache, model)
    st.success(f"Predicted Disease: {result}")
