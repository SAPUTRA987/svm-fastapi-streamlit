import streamlit as st
import pickle
import numpy as np

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# ==========================
# Load Model
# ==========================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ==========================
# Judul
# ==========================
st.title("🩺 Breast Cancer Prediction Using SVM")
st.markdown("---")

st.write("""
Masukkan nilai dari **30 fitur** di bawah ini, kemudian klik tombol **Predict**
untuk mengetahui hasil prediksi kanker payudara.
""")

# ==========================
# Nama Feature
# ==========================
features = [
    "Mean Radius",
    "Mean Texture",
    "Mean Perimeter",
    "Mean Area",
    "Mean Smoothness",
    "Mean Compactness",
    "Mean Concavity",
    "Mean Concave Points",
    "Mean Symmetry",
    "Mean Fractal Dimension",
    "Radius Error",
    "Texture Error",
    "Perimeter Error",
    "Area Error",
    "Smoothness Error",
    "Compactness Error",
    "Concavity Error",
    "Concave Points Error",
    "Symmetry Error",
    "Fractal Dimension Error",
    "Worst Radius",
    "Worst Texture",
    "Worst Perimeter",
    "Worst Area",
    "Worst Smoothness",
    "Worst Compactness",
    "Worst Concavity",
    "Worst Concave Points",
    "Worst Symmetry",
    "Worst Fractal Dimension"
]

# ==========================
# Input Data
# ==========================
inputs = []

col1, col2 = st.columns(2)

for i, feature in enumerate(features):

    if i % 2 == 0:
        with col1:
            value = st.number_input(feature, value=0.0, format="%.4f")
    else:
        with col2:
            value = st.number_input(feature, value=0.0, format="%.4f")

    inputs.append(value)

st.markdown("---")

# ==========================
# Prediksi
# ==========================
if st.button("🔍 Predict", use_container_width=True):

    try:

        x = np.array(inputs).reshape(1, -1)
        x = scaler.transform(x)

        prediction = model.predict(x)[0]
        probability = model.predict_proba(x).max()

        st.success("Prediction Completed")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:
                st.error("🔴 Prediction : Malignant (Cancer)")
            else:
                st.success("🟢 Prediction : Benign")

        with col2:
            st.metric(
                label="Prediction Probability",
                value=f"{probability:.2%}"
            )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

st.markdown("---")
st.caption("Breast Cancer Prediction System | Support Vector Machine (SVM)")