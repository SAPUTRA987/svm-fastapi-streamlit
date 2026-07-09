import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="SVM Prediction",
    layout="wide"
)

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Breast Cancer Prediction")
st.write("Input 30 Feature")

inputs = []

for i in range(30):
    value = st.number_input(f"Feature {i+1}")
    inputs.append(value)

if st.button("Predict"):

    x = np.array(inputs).reshape(1, -1)
    x = scaler.transform(x)

    prediction = model.predict(x)[0]
    probability = model.predict_proba(x).max()

    st.success(f"Prediction : {prediction}")
    st.metric("Probability", f"{probability:.2%}")