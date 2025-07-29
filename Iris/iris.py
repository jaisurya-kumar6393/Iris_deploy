# app.py
import streamlit as st
import numpy as np
import pickle



with open("Iris.pkl", "rb") as f:
    features = pickle.load(f)

species_names = ["setosa", "versi", "vergi"]

st.title("🌸 Iris Flower Species Predictor")
st.write("Enter the flower's dimensions below:")

# Input fields
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.5)
petal_width = st.slider("Petal Width (cm)", 0.1, 3.0, 0.2)

if st.button("Predict Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = features.predict(input_data)[0]
    st.success(f"🌼 Predicted Species: **{species_names[pred].capitalize()}**")
